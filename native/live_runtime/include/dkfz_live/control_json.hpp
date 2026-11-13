// Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Developed at the German Cancer Research Center (DKFZ).
// A bounded JSON value/parser for the standalone native control utility.
// No dependency on Python, Qt, TensorRT or a third party JSON library.
#pragma once
#include <cctype>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dkfz::live::control
{
struct Json
{
    enum Kind
    {
        null,
        boolean,
        number,
        string,
        object,
        array
    } kind = null;
    std::string scalar;
    std::map<std::string, Json> fields;
    std::vector<Json> elements;
    Json() = default;
    explicit Json(bool b) : kind(boolean), scalar(b ? "true" : "false") {}
    explicit Json(double n) : kind(number)
    {
        if (!std::isfinite(n))
            throw std::runtime_error("Non-finite JSON number");
        std::ostringstream out;
        out.precision(17);
        out << n;
        scalar = out.str();
    }
    explicit Json(std::string s) : kind(string), scalar(std::move(s)) {}
    explicit Json(const char *s) : Json(std::string(s)) {}
    static Json obj()
    {
        Json j;
        j.kind = object;
        return j;
    }
    const Json &at(const std::string &key) const
    {
        static const Json empty;
        auto it = fields.find(key);
        return it == fields.end() ? empty : it->second;
    }
    std::string text(const std::string &fallback = "") const { return kind == string ? scalar : fallback; }
    bool flag(bool fallback = false) const { return kind == boolean ? scalar == "true" : fallback; }
    double numeric(double fallback = 0) const { return kind == number ? std::stod(scalar) : fallback; }
    static std::string quote(const std::string &text)
    {
        std::string result = "\"";
        const char *hex = "0123456789abcdef";
        for (unsigned char c : text)
        {
            if (c == '"' || c == '\\')
            {
                result += '\\';
                result += static_cast<char>(c);
            }
            else if (c < 32)
            {
                result += "\\u00";
                result += hex[c >> 4];
                result += hex[c & 15];
            }
            else
                result += static_cast<char>(c);
        }
        return result + '"';
    }
    std::string dump() const
    {
        if (kind == null)
            return "null";
        if (kind == string)
            return quote(scalar);
        if (kind == boolean || kind == number)
            return scalar;
        std::string result = kind == object ? "{" : "[";
        bool first = true;
        if (kind == object)
            for (const auto &entry : fields)
            {
                if (!first)
                    result += ',';
                first = false;
                result += quote(entry.first) + ':' + entry.second.dump();
            }
        else
            for (const auto &entry : elements)
            {
                if (!first)
                    result += ',';
                first = false;
                result += entry.dump();
            }
        return result + (kind == object ? '}' : ']');
    }
};
class JsonParser
{
    const std::string &text_;
    std::size_t at_ = 0;
    void ws()
    {
        while (at_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[at_])))
            ++at_;
    }
    char take()
    {
        if (at_ == text_.size())
            throw std::runtime_error("Truncated JSON");
        return text_[at_++];
    }
    void expect(char c)
    {
        if (take() != c)
            throw std::runtime_error("Malformed JSON");
    }
    unsigned hex()
    {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const auto c = take();
            const auto digit = c >= '0' && c <= '9'   ? c - '0'
                               : c >= 'a' && c <= 'f' ? c - 'a' + 10
                               : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                      : -1;
            if (digit < 0)
                throw std::runtime_error("Invalid JSON unicode");
            value = value * 16 + static_cast<unsigned>(digit);
        }
        return value;
    }
    std::string str()
    {
        expect('"');
        std::string result;
        for (;;)
        {
            auto c = take();
            if (c == '"')
                return result;
            if (static_cast<unsigned char>(c) < 32)
                throw std::runtime_error("Invalid JSON string");
            if (c != '\\')
            {
                result += c;
                continue;
            }
            c = take();
            if (c == 'u')
            {
                unsigned cp = hex();
                if (cp >= 0xd800 && cp <= 0xdbff)
                {
                    expect('\\');
                    expect('u');
                    const auto low = hex();
                    if (low < 0xdc00 || low > 0xdfff)
                        throw std::runtime_error("Invalid JSON surrogate");
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                }
                else if (cp >= 0xdc00 && cp <= 0xdfff)
                    throw std::runtime_error("Invalid JSON surrogate");
                if (cp < 0x80)
                    result += static_cast<char>(cp);
                else
                {
                    if (cp >= 0x10000)
                        result += static_cast<char>(0xf0 | (cp >> 18));
                    if (cp >= 0x800)
                        result += static_cast<char>((cp >= 0x10000 ? 0x80 : 0xe0) |
                                                    ((cp >> 12) & (cp >= 0x10000 ? 0x3f : 0x0f)));
                    result += static_cast<char>((cp >= 0x800 ? 0x80 : 0xc0) |
                                                ((cp >> 6) & (cp >= 0x800 ? 0x3f : 0x1f)));
                    result += static_cast<char>(0x80 | (cp & 0x3f));
                }
            }
            else
            {
                const std::string escapes = "\"\\/bfnrt";
                const std::string values = "\"\\/\b\f\n\r\t";
                const auto index = escapes.find(c);
                if (index == std::string::npos)
                    throw std::runtime_error("Invalid JSON escape");
                result += values[index];
            }
        }
    }
    Json value(unsigned depth)
    {
        if (depth > 64)
            throw std::runtime_error("JSON nesting too deep");
        ws();
        if (at_ == text_.size())
            throw std::runtime_error("Missing JSON value");
        const auto c = text_[at_];
        if (c == '"')
            return Json(str());
        if (c == '{' || c == '[')
        {
            ++at_;
            Json j;
            j.kind = c == '{' ? Json::object : Json::array;
            const char end = c == '{' ? '}' : ']';
            ws();
            if (at_ < text_.size() && text_[at_] == end)
            {
                ++at_;
                return j;
            }
            for (;;)
            {
                ws();
                std::string key;
                if (c == '{')
                {
                    key = str();
                    ws();
                    expect(':');
                }
                auto v = value(depth + 1);
                if (c == '{')
                {
                    if (!j.fields.emplace(key, std::move(v)).second)
                        throw std::runtime_error("Duplicate JSON key");
                }
                else
                    j.elements.push_back(std::move(v));
                ws();
                const auto separator = take();
                if (separator == end)
                    return j;
                if (separator != ',')
                    throw std::runtime_error("Malformed JSON separator");
            }
        }
        for (const std::string literal : {"true", "false", "null"})
        {
            if (text_.compare(at_, literal.size(), literal) == 0)
            {
                at_ += literal.size();
                return literal == "null" ? Json() : Json(literal == "true");
            }
        }
        const auto begin = at_;
        if (text_[at_] == '-')
            ++at_;
        if (at_ == text_.size())
            throw std::runtime_error("Invalid JSON number");
        if (text_[at_] == '0')
            ++at_;
        else
        {
            if (text_[at_] < '1' || text_[at_] > '9')
                throw std::runtime_error("Invalid JSON value");
            while (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_])))
                ++at_;
        }
        for (const char part : {'.', 'e'})
        {
            if (at_ < text_.size() && (text_[at_] == part || (part == 'e' && text_[at_] == 'E')))
            {
                ++at_;
                if (part == 'e' && at_ < text_.size() && (text_[at_] == '+' || text_[at_] == '-'))
                    ++at_;
                const auto digits = at_;
                while (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_])))
                    ++at_;
                if (digits == at_)
                    throw std::runtime_error("Invalid JSON number");
            }
        }
        Json j;
        j.kind = Json::number;
        j.scalar = text_.substr(begin, at_ - begin);
        if (!std::isfinite(std::stod(j.scalar)))
            throw std::runtime_error("Non-finite JSON number");
        return j;
    }

  public:
    explicit JsonParser(const std::string &text) : text_(text) {}
    Json parse()
    {
        auto result = value(0);
        ws();
        if (at_ != text_.size())
            throw std::runtime_error("Trailing JSON data");
        return result;
    }
};
inline Json parse(const std::string &text) { return JsonParser(text).parse(); }
} // namespace dkfz::live::control
