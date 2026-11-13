// Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Standalone managed-service recovery control. Standard C++/Linux only; no Python.
#include "dkfz_live/control_json.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using dkfz::live::control::Json;
using dkfz::live::control::parse;
using Settings = std::map<std::string, std::string>;
namespace
{
constexpr auto service = "dkfz-native-uart-live.service";
constexpr auto schema = "dkfz-live-recovery-control-v1";
std::string trim(const std::string &s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string quote(const std::string &s)
{
    std::string out = "'";
    for (const auto c : s)
        out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
}
std::string command(const std::string &s)
{
    auto *pipe = popen(s.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("Cannot execute host query");
    std::string result;
    char buf[4096];
    while (const auto n = fread(buf, 1, sizeof buf, pipe))
    {
        result.append(buf, n);
        if (result.size() > 8 * 1024 * 1024)
        {
            pclose(pipe);
            throw std::runtime_error("Host query too large");
        }
    }
    if (pclose(pipe) != 0)
        throw std::runtime_error("Host query failed: " + s);
    return result;
}
std::string read(const fs::path &path, std::size_t limit = 2097152)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read " + path.string());
    std::string text(limit + 1, '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(input.gcount()));
    if (input.bad() || text.size() > limit)
        throw std::runtime_error("Oversized/unreadable " + path.string());
    return text;
}
Settings settings(const std::string &text)
{
    Settings out;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
    {
        line = trim(line.substr(0, line.find('#')));
        if (line.empty())
            continue;
        const auto at = line.find('=');
        if (at == std::string::npos || at == 0 ||
            !out.emplace(trim(line.substr(0, at)), trim(line.substr(at + 1))).second)
            throw std::runtime_error("Malformed/duplicate setting");
    }
    return out;
}
std::string sha(const fs::path &path)
{
    return command("sha256sum -- " + quote(path.string())).substr(0, 64);
}
std::string stable_read(const fs::path &path, std::string &revision, std::size_t limit = 2097152)
{
    revision = sha(path);
    const auto text = read(path, limit);
    if (sha(path) != revision)
        throw std::runtime_error("File changed during read; refresh required");
    return text;
}
std::string invocation()
{
    return trim(command(std::string("systemctl show --value -p InvocationID ") + service));
}
Json identity(const Settings &m)
{
    auto j = Json::obj();
    for (const auto *key : {"model_name", "model_variant", "engine_size_bytes", "engine_crc32"})
        j.fields[key] = Json(m.at(key));
    if (!std::regex_match(m.at("engine_crc32"), std::regex("[0-9a-f]{8}")) ||
        !std::regex_match(m.at("engine_size_bytes"), std::regex("[0-9]+")) ||
        std::stoull(m.at("engine_size_bytes")) == 0)
        throw std::runtime_error("Missing/invalid engine fingerprint");
    j.fields["engine_size_bytes"] = Json(std::to_string(std::stoull(m.at("engine_size_bytes"))));
    return j;
}
std::optional<double> recovery_number(const std::string &text)
{
    if (text.empty())
        return {};
    std::size_t used = 0;
    const auto value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value <= 0 || value > 1)
        throw std::runtime_error("Recovery threshold must be finite within (0, 1]");
    return value;
}
Json optional(std::optional<double> value) { return value ? Json(*value) : Json(); }
Json policy(bool enabled, std::optional<double> threshold)
{
    auto j = Json::obj();
    j.fields["enabled"] = Json(enabled);
    j.fields["threshold"] = enabled ? optional(threshold) : Json();
    return j;
}
fs::path mapped(const Settings &properties, const std::string &path)
{
    auto wd = properties.at("WorkingDirectory");
    auto target =
        (fs::path(path).is_absolute() ? fs::path(path) : fs::path(wd) / path).lexically_normal().string();
    auto result = target;
    std::size_t longest = 0;
    std::istringstream bindings(properties.at("BindReadOnlyPaths"));
    std::string bind;
    while (bindings >> bind)
    {
        if (bind.find('\\') != std::string::npos)
            throw std::runtime_error("Unsupported systemd binding");
        if (bind.front() == '-')
            bind.erase(0, 1);
        const auto at = bind.find(':');
        const auto src = bind.substr(0, at);
        auto dst = at == std::string::npos ? src : bind.substr(at + 1);
        dst = dst.substr(0, dst.find(':'));
        if (src.empty() || dst.empty() || src.front() != '/' || dst.front() != '/')
            throw std::runtime_error("Invalid systemd binding");
        if ((target == dst || target.rfind(dst + "/", 0) == 0) && dst.size() > longest)
        {
            result = src + target.substr(dst.size());
            longest = dst.size();
        }
    }
    return fs::path(result).lexically_normal();
}
struct Snapshot
{
    Settings props, config, manifest, saved;
    fs::path config_path, manifest_path, override_path, source_path, primary_path;
    std::string invocation_id, revision, request_id = "0";
    std::string config_revision, manifest_revision,
        source_revision = "missing", override_revision = "missing", primary_revision = "missing";
    Json id, source, active = Json::obj(), diagnostic;
    bool model_support = false, source_available = false, source_unchanged = false;
    bool enabled = false, live = false, paused = false;
    std::optional<double> selected, calibrated, effective;
    std::string reason;
    std::string file_revision() const
    {
        return sha(config_path) + ":" + sha(manifest_path) + ":" +
               (fs::exists(primary_path) ? sha(primary_path) : "missing") + ":" +
               (fs::exists(override_path) ? sha(override_path) : "missing") + ":" +
               (!source_path.empty() && fs::exists(source_path) ? sha(source_path) : "missing") + ":" +
               invocation_id;
    }
    Json response(const std::string &confirmation) const
    {
        auto j = Json::obj();
        auto primary = Json::obj();
        for (const auto *field : {"threshold", "early_emission", "emission_threshold"})
            primary.fields[field] = diagnostic.at(field);
        j.fields = {{"primary_policy", primary},
                    {"schema", Json(schema)},
                    {"api_version", Json(1.)},
                    {"identity", id},
                    {"model_supports_recovery", Json(model_support)},
                    {"live_api_supported", Json(live)},
                    {"reason", Json(reason)},
                    {"calibrated_threshold", optional(calibrated)},
                    {"source_revision", Json(source_revision)},
                    {"source_metadata_available", Json(source_available)},
                    {"source_predates_engine", Json(source_unchanged)},
                    {"source_recovery_enabled",
                     source_available ? Json(source.at("recovery_enabled").flag(true)) : Json()},
                    {"source_default_enabled",
                     source_available
                         ? Json(source.at("recovery_head").flag() &&
                                source.at("recovery_enabled").flag(true) && calibrated.has_value())
                         : Json()},
                    {"selected_threshold", optional(selected)},
                    {"requested", policy(enabled, effective)},
                    {"active", active},
                    {"request_id", Json(request_id)},
                    {"service_invocation_id", Json(invocation_id)},
                    {"confirmation", Json(confirmation)},
                    {"input_paused", Json(paused)},
                    {"revision", Json(revision)}};
        return j;
    }
};
Snapshot snapshot(bool read_journal = true)
{
    Snapshot s;
    s.props = settings(command(std::string("systemctl show ") + service +
                               " -p WorkingDirectory -p BindReadOnlyPaths -p RootDirectory -p InvocationID "
                               "-p MainPID -p LogNamespace -p NeedDaemonReload"));
    s.invocation_id = s.props["InvocationID"];
    if (!std::regex_match(s.invocation_id, std::regex("[0-9a-f]{32}")) || s.props["MainPID"].empty() ||
        s.props["MainPID"] == "0")
        throw std::runtime_error(
            "Live recovery requires the running managed service; direct processes are unsupported");
    if (!s.props["RootDirectory"].empty() || s.props["NeedDaemonReload"] != "no" ||
        s.props["WorkingDirectory"].empty() || s.props["WorkingDirectory"].front() != '/' ||
        fs::canonical(mapped(s.props, s.props["WorkingDirectory"])) != fs::canonical(fs::current_path()))
        throw std::runtime_error("Service must use this remote project and current unit definition");
    if (!std::regex_match(s.props["MainPID"], std::regex("[0-9]+")))
        throw std::runtime_error("Invalid service PID");
    auto argv = read(fs::path("/proc") / s.props["MainPID"] / "cmdline", 65536);
    std::vector<std::string> args;
    std::size_t at = 0;
    while (at < argv.size())
    {
        const auto end = argv.find('\0', at);
        args.push_back(argv.substr(at, end - at));
        if (end == std::string::npos)
            break;
        at = end + 1;
    }
    std::string config;
    for (std::size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "--config" && i + 1 < args.size())
            config = args[++i];
        else if (args[i].rfind("--config=", 0) == 0)
            config = args[i].substr(9);
        else if (args[i].rfind("--manifest", 0) == 0 || args[i].rfind("--threshold", 0) == 0 ||
                 args[i].rfind("--early-emission", 0) == 0 || args[i] == "--no-early-emission")
            throw std::runtime_error("Command-line policy overrides unsupported");
    }
    if (config.empty())
        throw std::runtime_error("Cannot resolve managed service config");
    s.config_path = mapped(s.props, config);
    s.config = settings(stable_read(s.config_path, s.config_revision));
    s.manifest_path = mapped(s.props, s.config.at("manifest_path"));
    s.manifest = settings(stable_read(s.manifest_path, s.manifest_revision));
    s.id = identity(s.manifest);
    s.model_support = s.manifest["recovery_head"] == "true";
    const auto primary_path = s.config["threshold_override_path"];
    if (primary_path.empty())
        throw std::runtime_error("Runtime live policy mechanism is disabled");
    s.primary_path = mapped(s.props, primary_path);
    if (fs::exists(s.primary_path))
        s.primary_revision = sha(s.primary_path);
    s.override_path = fs::path(s.primary_path.string() + ".recovery");
    if (fs::is_symlink(s.override_path))
        throw std::runtime_error("Refusing recovery override symlink");
    s.selected = recovery_number(s.manifest["recovery_threshold"]);
    s.enabled = s.model_support && s.selected.has_value();
    s.effective = s.enabled ? s.selected : std::nullopt;
    try
    {
        const auto dir = fs::weakly_canonical(fs::current_path() / s.manifest.at("source_model_dir"));
        const auto root = fs::canonical(fs::current_path()).string() + "/";
        if (dir.string().rfind(root, 0) != 0)
            throw std::runtime_error("Source outside remote checkout");
        const auto variant = s.manifest.at("model_variant");
        if (variant != "best" && variant != "tuned")
            throw std::runtime_error("Unknown source variant");
        const auto source = fs::canonical(dir / (variant == "best" ? "train_config.json" : "tuning/tuned_config.json"));
        if (source.string().rfind(root, 0) != 0)
            throw std::runtime_error("Source config outside remote checkout");
        s.source_path = source;
        s.source = parse(stable_read(s.source_path, s.source_revision));
        s.source_available = s.source.kind == Json::object &&
                             s.source.at("modelname").text() == s.manifest["model_name"] &&
                             s.source.at("recovery_head").flag() == s.model_support;
        if (s.source_available && s.source.at("recovery_threshold").kind == Json::number)
            s.calibrated = recovery_number(s.source.at("recovery_threshold").scalar);
        const fs::path engine_value = s.manifest.at("engine_path");
        // Match load_deployment_manifest: relative engine paths are relative
        // to model.conf, not to the runtime/project working directory.
        const auto engine = engine_value.is_absolute() ? mapped(s.props, engine_value.string())
                                                       : s.manifest_path.parent_path() / engine_value;
        s.source_unchanged = fs::last_write_time(s.source_path) <= fs::last_write_time(engine);
    }
    catch (const std::exception &)
    {
        s.source_available = false;
        s.source_unchanged = false;
        s.calibrated = {};
    }
    if (fs::exists(s.override_path))
    {
        s.saved = settings(stable_read(s.override_path, s.override_revision, 2048));
        bool same = s.saved["schema"] == "dkfz-live-recovery-override-v1";
        if (!same || s.saved.size() != 9)
            throw std::runtime_error("Malformed recovery override");
        for (const auto &entry : s.id.fields)
            same = same && s.saved[entry.first] == entry.second.text();
        if (same && s.model_support)
        {
            // Preserve the last selected threshold across disable and service restarts.
            s.selected = recovery_number(s.saved["selected_threshold"]);
            if (s.saved["service_invocation_id"] == s.invocation_id)
            {
                if (s.saved["enabled"] != "true" && s.saved["enabled"] != "false")
                    throw std::runtime_error("Invalid saved enabled state");
                s.enabled = s.saved["enabled"] == "true";
                s.effective = s.enabled ? s.selected : std::nullopt;
                s.request_id = s.saved["request_id"];
                if (!std::regex_match(s.request_id, std::regex("[0-9]+")) ||
                    std::stoull(s.request_id) == 0 ||
                    std::stoull(s.request_id) == std::numeric_limits<std::uint64_t>::max())
                    throw std::runtime_error("Invalid saved request ID");
                if (s.enabled && !s.selected)
                    throw std::runtime_error("Recovery requires two outputs and an explicit threshold");
            }
        }
    }
    s.revision = s.config_revision + ":" + s.manifest_revision + ":" + s.primary_revision + ":" +
                 s.override_revision + ":" + s.source_revision + ":" + s.invocation_id;
    if (read_journal)
    {
        const auto ns = s.props["LogNamespace"];
        const auto records = command(std::string("journalctl --no-pager --output=json --unit=") + service +
                                     (ns.empty() ? "" : " --namespace=" + quote(ns)) +
                                     " _SYSTEMD_INVOCATION_ID=" + quote(s.invocation_id) + " --lines=300");
        std::istringstream lines(records);
        std::string line;
        std::uint64_t newest = 0;
        const auto now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                        std::chrono::system_clock::now().time_since_epoch())
                                                        .count());
        const auto interval = [&](const char *key, std::uint64_t fallback)
        {
            const auto value = s.config[key];
            return value.empty() ? fallback : std::stoull(value);
        };
        const auto freshness =
            std::max<std::uint64_t>(15000000, (2 * std::max(interval("status_interval_ms", 5000),
                                                            interval("paused_status_interval_ms", 60000)) +
                                               10000) *
                                                  1000);
        while (std::getline(lines, line))
            try
            {
                const auto entry = parse(line);
                const auto event = parse(entry.at("MESSAGE").text());
                const auto timestamp = std::stoull(entry.at("__REALTIME_TIMESTAMP").text());
                if (entry.at("_SYSTEMD_INVOCATION_ID").text() != s.invocation_id || timestamp > now ||
                    timestamp + freshness < now || timestamp < newest ||
                    event.at("schema").text() != "dkfz-live-diagnostic-v1" ||
                    event.at("recovery_api_version").numeric() != 1 ||
                    event.at("service_invocation_id").text() != s.invocation_id ||
                    (event.at("code").text() != "runtime_status" &&
                     event.at("code").text() != "recovery_applied"))
                    continue;
                bool same = true;
                for (const auto &item : s.id.fields)
                    same = same && event.at(item.first).text() == item.second.text();
                if (!same || event.at("active_recovery_enabled").kind != Json::boolean ||
                    event.at("active_recovery_threshold").kind != Json::number ||
                    event.at("recovery_request_id").kind != Json::string)
                    continue;
                const bool enabled = event.at("active_recovery_enabled").flag();
                const auto number = event.at("active_recovery_threshold").numeric();
                if ((enabled && (number <= 0 || number > 1)) || (!enabled && number != -1))
                    continue;
                newest = timestamp;
                s.live = true;
                s.diagnostic = event;
                s.active = policy(enabled, enabled ? std::optional<double>(number) : std::nullopt);
                s.active.fields["request_id"] = event.at("recovery_request_id");
                s.paused = event.at("input_paused").flag();
            }
            catch (const std::exception &)
            { /* Ignore unrelated/unusable journal records. */
            }
    }
    if (!s.live)
        s.reason = "Live switching unavailable: no fresh identity-bound native API v1 diagnostics "
                   "from the current service invocation. Use stop / prepare / validate / start. "
                   "Direct sessions are unsupported.";
    if (invocation() != s.invocation_id || s.file_revision() != s.revision)
        throw std::runtime_error("Deployment/invocation/revision changed during read; refresh required");
    return s;
}
void write_atomic(const fs::path &path, const std::string &text)
{
    std::string name = path.string() + ".XXXXXX";
    std::vector<char> temp(name.begin(), name.end());
    temp.push_back(0);
    const int fd = mkstemp(temp.data());
    if (fd < 0)
        throw std::runtime_error("Cannot create recovery override");
    bool closed = false;
    try
    {
        std::size_t offset = 0;
        while (offset < text.size())
        {
            const auto n = ::write(fd, text.data() + offset, text.size() - offset);
            if (n <= 0)
                throw std::runtime_error("Recovery write failed");
            offset += static_cast<std::size_t>(n);
        }
        if (fchmod(fd, 0644) != 0 || fsync(fd) != 0)
            throw std::runtime_error("Recovery durability failed");
        close(fd);
        closed = true;
        if (rename(temp.data(), path.c_str()) != 0)
            throw std::runtime_error("Recovery atomic replacement failed");
        const int dir = open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
        if (dir < 0)
            throw std::runtime_error("Cannot sync recovery directory");
        const int result = fsync(dir);
        close(dir);
        if (result != 0)
            throw std::runtime_error("Recovery directory sync failed; request may have been saved");
    }
    catch (...)
    {
        if (!closed)
            close(fd);
        unlink(temp.data());
        throw;
    }
}
} // namespace
int main(int argc, char **argv)
{
    try
    {
        std::string action, expected_revision, expected_invocation;
        std::optional<double> custom;
        bool json = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--json")
                json = true;
            else if (arg == "--status" || arg == "--enable" || arg == "--disable" ||
                     arg == "--restore-calibrated")
            {
                if (!action.empty())
                    throw std::runtime_error("Choose one recovery action");
                action = arg;
            }
            else if ((arg == "--threshold" || arg == "--expected-revision" ||
                      arg == "--expected-invocation") &&
                     i + 1 < argc)
            {
                const std::string value = argv[++i];
                if (arg == "--threshold")
                    custom = recovery_number(value);
                else if (arg == "--expected-revision")
                    expected_revision = value;
                else
                    expected_invocation = value;
            }
            else
                throw std::runtime_error("Unknown/incomplete argument: " + arg);
        }
        if (!json)
            throw std::runtime_error("Use --json; native recovery API schema v1");
        if (action.empty() && !custom)
            throw std::runtime_error("Choose a recovery action or threshold");
        if ((action == "--status" || action == "--restore-calibrated") && custom)
            throw std::runtime_error("Threshold cannot accompany status/restore");
        int lock =
            open(".jetson-control-deployment.lock", O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0644);
        if (lock >= 0)
        {
            // sudo must not leave a root-owned lock that later blocks the SSH
            // user's preparation workflow. Match the remote project owner.
            struct stat project{};
            if (::stat(".", &project) != 0 ||
                (geteuid() == 0 && fchown(lock, project.st_uid, project.st_gid) != 0))
                throw std::runtime_error("Cannot assign deployment lock ownership");
        }
        else if (errno == EEXIST)
        {
            lock = open(".jetson-control-deployment.lock", O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (lock < 0 || flock(lock, LOCK_EX) != 0)
            throw std::runtime_error("Cannot lock deployment");
        auto s = snapshot();
        if (action == "--status")
        {
            const bool confirmed =
                s.live && s.active.at("request_id").text() == s.request_id &&
                s.active.at("enabled").flag() == s.enabled &&
                s.active.at("threshold").dump() == (s.enabled ? optional(s.effective).dump() : "null");
            std::cout << s.response(confirmed  ? "confirmed"
                                    : s.paused ? "pending — input paused"
                                               : "unconfirmed / pending")
                             .dump()
                      << '\n';
            return 0;
        }
        if (!s.live || !s.model_support)
            throw std::runtime_error(!s.model_support ? "Recovery unavailable: single-output model"
                                                      : s.reason);
        if ((!expected_revision.empty() && expected_revision != s.revision) ||
            (!expected_invocation.empty() && expected_invocation != s.invocation_id))
            throw std::runtime_error("Stale deployment/invocation/revision; Refresh required");
        if (action == "--restore-calibrated")
        {
            if (!s.source_available || !s.source_unchanged)
                throw std::runtime_error("Source calibration unavailable or changed since deployment");
            s.selected = s.calibrated;
            s.enabled = s.source.at("recovery_head").flag() && s.source.at("recovery_enabled").flag(true) &&
                        s.selected.has_value();
        }
        else
        {
            if (custom)
                s.selected = custom;
            if (action == "--enable")
            {
                if (!s.selected && s.source_available && s.source_unchanged)
                    s.selected = s.calibrated;
                if (!s.selected)
                    throw std::runtime_error(
                        "No saved or calibrated threshold; explicit valid custom threshold required");
                s.enabled = true;
            }
            else if (action == "--disable")
                s.enabled = false;
        }
        if (s.enabled && !s.selected)
            throw std::runtime_error("Enabled recovery requires selected threshold");
        s.effective = s.enabled ? s.selected : std::nullopt;
        auto token = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::system_clock::now().time_since_epoch())
                                                    .count());
        token = std::max<std::uint64_t>(token, std::stoull(s.request_id) + std::uint64_t{1});
        s.request_id = std::to_string(token);
        std::string text = "schema=dkfz-live-recovery-override-v1\n";
        for (const auto &entry : s.id.fields)
        {
            if (entry.second.text().find_first_of("\r\n#") != std::string::npos)
                throw std::runtime_error("Malformed deployment identity");
            text += entry.first + "=" + entry.second.text() + "\n";
        }
        text += "service_invocation_id=" + s.invocation_id + "\nrequest_id=" + s.request_id +
                "\nenabled=" + (s.enabled ? "true" : "false") +
                "\nselected_threshold=" + (s.selected ? optional(s.selected).dump() : "") + "\n";
        if (text.size() > 2048)
            throw std::runtime_error("Recovery override too large");
        if (s.file_revision() != s.revision || invocation() != s.invocation_id)
            throw std::runtime_error("Revision/invocation changed before save");
        write_atomic(s.override_path, text);
        s.revision = s.file_revision();
        s.active = Json::obj();
        std::cout << s.response("saved/requested — pending confirmation").dump() << '\n';
        close(lock);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Recovery control: " << error.what() << '\n';
        return 2;
    }
}
