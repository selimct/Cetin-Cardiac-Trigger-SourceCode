/*
 * Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * Developed at the German Cancer Research Center (DKFZ).
 */

#include "dkfz_live/live_core.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace dkfz::live;

namespace
{

template <typename T>
T value_or(const py::dict &values, const char *key, T fallback)
{
    if (!values.contains(key) || values[key].is_none())
    {
        return fallback;
    }
    return values[key].cast<T>();
}

std::vector<std::string> string_vector_or(
    const py::dict &values,
    const char *key,
    std::vector<std::string> fallback = {})
{
    if (!values.contains(key) || values[key].is_none())
    {
        return fallback;
    }
    return values[key].cast<std::vector<std::string>>();
}

SignalQualityStats signal_quality_stats_from_python(const py::object &value)
{
    SignalQualityStats stats;
    if (value.is_none())
    {
        return stats;
    }

    const auto values = value.cast<py::dict>();
    stats.present = true;
    stats.transform = value_or<std::string>(values, "transform", "");
    stats.ecg_component_cols = string_vector_or(
        values,
        "ecg_component_cols",
        ecg_quality_component_names());
    stats.puls_component_cols = string_vector_or(values, "puls_component_cols");
    stats.median = value_or<std::map<std::string, double>>(
        values,
        "median",
        {});
    stats.iqr = value_or<std::map<std::string, double>>(values, "iqr", {});
    stats.min_iqr = value_or<std::map<std::string, double>>(
        values,
        "min_iqr",
        {});
    if (values.contains("z_clip") && !values["z_clip"].is_none())
    {
        stats.z_clip = values["z_clip"].cast<double>();
    }
    stats.ecg_weight = value_or<double>(values, "ecg_weight", 1.0);
    stats.puls_weight = value_or<double>(values, "puls_weight", 0.0);
    stats.eps = value_or<double>(values, "eps", 1e-6);
    return stats;
}

FeatureFramerConfig feature_config_from_python(const py::dict &values)
{
    FeatureFramerConfig config;
    config.signal_quality_puls_flat_window_frames = value_or<std::size_t>(values, "signal_quality_puls_flat_window_frames", 10);
    config.signal_quality_puls_median_window_frames = value_or<std::size_t>(values, "signal_quality_puls_median_window_frames", 200);
    config.reset_on_discontinuity = value_or<bool>(values, "reset_on_discontinuity", false);
    config.raw_hz = value_or<double>(values, "raw_hz", config.raw_hz);
    config.frame_size_raw = value_or<std::size_t>(
        values,
        "frame_size_raw",
        config.frame_size_raw);
    config.include_puls = value_or<bool>(
        values,
        "include_puls",
        config.include_puls);
    config.ecg1_offset = value_or<double>(
        values,
        "ecg1_offset",
        config.ecg1_offset);
    config.ecg2_offset = value_or<double>(
        values,
        "ecg2_offset",
        config.ecg2_offset);
    config.ecg3_offset = value_or<double>(
        values,
        "ecg3_offset",
        config.ecg3_offset);
    config.include_ecg3 = value_or<bool>(
        values,
        "include_ecg3",
        config.include_ecg3);
    config.ecg_slope_mode = parse_ecg_slope_mode(value_or<std::string>(
        values,
        "ecg_slope_mode",
        "none"));
    config.ecg_slope_slow_window_samples = value_or<std::size_t>(
        values,
        "ecg_slope_slow_window_samples",
        config.ecg_slope_slow_window_samples);
    config.signal_quality_mode = parse_signal_quality_mode(value_or<std::string>(
        values,
        "signal_quality_mode",
        "none"));
    if (values.contains("signal_quality_stats"))
    {
        config.signal_quality_stats = signal_quality_stats_from_python(
            values["signal_quality_stats"]);
    }
    config.ecg_rotation_angle = value_or<double>(
        values,
        "ecg_rotation_angle",
        config.ecg_rotation_angle);
    config.puls_offset = value_or<double>(
        values,
        "puls_offset",
        config.puls_offset);
    config.ecg_quantile = value_or<double>(
        values,
        "ecg_quantile",
        config.ecg_quantile);
    config.ecg_norm_window_samples = value_or<std::size_t>(
        values,
        "ecg_norm_window_samples",
        config.ecg_norm_window_samples);
    config.ecg_norm_eps = value_or<double>(
        values,
        "ecg_norm_eps",
        config.ecg_norm_eps);
    config.ecg_norm_warmup_scale = value_or<double>(
        values,
        "ecg_norm_warmup_scale",
        config.ecg_norm_warmup_scale);
    config.puls_quantile_low = value_or<double>(
        values,
        "puls_quantile_low",
        config.puls_quantile_low);
    config.puls_quantile_high = value_or<double>(
        values,
        "puls_quantile_high",
        config.puls_quantile_high);
    config.puls_norm_window_frames = value_or<std::size_t>(
        values,
        "puls_norm_window_frames",
        config.puls_norm_window_frames);
    config.puls_norm_eps = value_or<double>(
        values,
        "puls_norm_eps",
        config.puls_norm_eps);
    config.puls_norm_warmup_value = value_or<double>(
        values,
        "puls_norm_warmup_value",
        config.puls_norm_warmup_value);
    config.puls_clip = value_or<bool>(
        values,
        "puls_clip",
        config.puls_clip);
    config.align_to_puls_blocks = value_or<bool>(
        values,
        "align_to_puls_blocks",
        config.align_to_puls_blocks);
    config.verify_puls_constant_within_block = value_or<bool>(
        values,
        "verify_puls_constant_within_block",
        config.verify_puls_constant_within_block);
    config.puls_constant_tol = value_or<double>(
        values,
        "puls_constant_tol",
        config.puls_constant_tol);
    return config;
}

std::optional<double> optional_double(const py::object &value)
{
    return value.is_none()
        ? std::nullopt
        : std::optional<double>(value.cast<double>());
}

std::optional<std::int64_t> optional_int64(const py::object &value)
{
    return value.is_none()
        ? std::nullopt
        : std::optional<std::int64_t>(value.cast<std::int64_t>());
}

RawSample raw_sample_from_python(
    double ecg1,
    double ecg2,
    const py::object &puls,
    const py::object &sample_idx,
    const py::object &time_ms,
    const py::object &ecg3)
{
    return {
        ecg1,
        ecg2,
        optional_double(ecg3),
        optional_double(puls),
        optional_int64(sample_idx),
        optional_double(time_ms),
    };
}

py::array_t<float> features_as_array(const FeatureFrame &frame)
{
    py::array_t<float> result(frame.features.size());
    auto mutable_result = result.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < mutable_result.shape(0); ++index)
    {
        mutable_result(index) = frame.features[static_cast<std::size_t>(index)];
    }
    return result;
}

std::string sid_as_string(const py::object &sid)
{
    return py::str(sid).cast<std::string>();
}

template <typename T>
std::optional<std::vector<T>> optional_array_values(
    const py::object &value,
    std::size_t expected_size,
    const char *name)
{
    if (value.is_none())
    {
        return std::nullopt;
    }
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
        value);
    if (!array || array.ndim() != 1 ||
        static_cast<std::size_t>(array.shape(0)) != expected_size)
    {
        throw std::invalid_argument(
            std::string(name) + " must be a one-dimensional array with " +
            std::to_string(expected_size) + " values.");
    }
    const auto values = array.template unchecked<1>();
    std::vector<T> result(expected_size);
    for (std::size_t index = 0; index < expected_size; ++index)
    {
        result[index] = values(static_cast<py::ssize_t>(index));
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_live_core, module)
{
    module.doc() = "Canonical native live preprocessing and trigger core.";
    module.attr("PREPROCESSING_CONTRACT_VERSION") = kPreprocessingContractVersion;

    py::class_<FeatureFrame>(module, "NativeFeatureFrame")
        .def_property_readonly("frame_idx_50hz", [](const FeatureFrame &value)
        {
            return value.frame_idx;
        })
        .def_property_readonly("start_sample_idx_400hz", [](const FeatureFrame &value)
        {
            return value.start_sample_idx;
        })
        .def_property_readonly("end_sample_idx_400hz", [](const FeatureFrame &value)
        {
            return value.end_sample_idx;
        })
        .def_readonly("start_time_ms", &FeatureFrame::start_time_ms)
        .def_readonly("end_time_ms", &FeatureFrame::end_time_ms)
        .def_property_readonly("features", &features_as_array)
        .def_readonly("auxiliary_features", &FeatureFrame::auxiliary_features);

    py::class_<RollingPairNormQuantileNormalizer>(
        module,
        "NativeRollingPairNormQuantileNormalizer")
        .def(
            py::init<double, std::size_t, double, double>(),
            py::arg("q") = 0.999,
            py::arg("window_samples") = 4000,
            py::arg("eps") = 1e-8,
            py::arg("warmup_scale") = 1.0)
        .def("reset", &RollingPairNormQuantileNormalizer::reset)
        .def("step", &RollingPairNormQuantileNormalizer::step)
        .def(
            "step_with_extra",
            [](RollingPairNormQuantileNormalizer &normalizer,
               double ecg1,
               double ecg2,
               const py::object &ecg3)
            {
                const auto values = normalizer.step_with_extra(
                    ecg1,
                    ecg2,
                    optional_double(ecg3));
                return py::make_tuple(values[0], values[1], values[2]);
            },
            py::arg("ecg1"),
            py::arg("ecg2"),
            py::arg("ecg3") = py::none())
        .def_property_readonly("q", &RollingPairNormQuantileNormalizer::q)
        .def_property_readonly(
            "window_samples",
            &RollingPairNormQuantileNormalizer::window_samples)
        .def_property_readonly("eps", &RollingPairNormQuantileNormalizer::eps)
        .def_property_readonly(
            "warmup_scale",
            &RollingPairNormQuantileNormalizer::warmup_scale)
        .def_property_readonly(
            "last_scale",
            &RollingPairNormQuantileNormalizer::last_scale);

    py::class_<RollingScalarQuantileNormalizer>(
        module,
        "NativeRollingScalarQuantileNormalizer")
        .def(
            py::init<double, double, std::size_t, double, double, bool>(),
            py::arg("q_low") = 0.05,
            py::arg("q_high") = 0.95,
            py::arg("window_frames") = 500,
            py::arg("eps") = 1e-8,
            py::arg("warmup_value") = 0.5,
            py::arg("clip") = true)
        .def("reset", &RollingScalarQuantileNormalizer::reset)
        .def("step", &RollingScalarQuantileNormalizer::step)
        .def_property_readonly("q_low", &RollingScalarQuantileNormalizer::q_low)
        .def_property_readonly("q_high", &RollingScalarQuantileNormalizer::q_high)
        .def_property_readonly(
            "window_frames",
            &RollingScalarQuantileNormalizer::window_frames)
        .def_property_readonly("eps", &RollingScalarQuantileNormalizer::eps)
        .def_property_readonly(
            "warmup_value",
            &RollingScalarQuantileNormalizer::warmup_value)
        .def_property_readonly("clip", &RollingScalarQuantileNormalizer::clip);

    py::class_<SignalQualityTracker>(module, "NativeSignalQualityTracker")
        .def(
            py::init([](const std::string &mode,
                        const py::object &stats,
                        double eps,
                        double ecg_flat_threshold,
                        double puls_flat_threshold,
                        std::size_t puls_flat_window_frames,
                        std::size_t puls_median_window_frames)
            {
                return SignalQualityTracker(
                    parse_signal_quality_mode(mode),
                    signal_quality_stats_from_python(stats),
                    eps,
                    ecg_flat_threshold,
                    puls_flat_threshold,
                    puls_flat_window_frames,
                    puls_median_window_frames);
            }),
            py::arg("mode") = "none",
            py::arg("stats") = py::none(),
            py::arg("eps") = 1e-6,
            py::arg("ecg_flat_threshold") = 0.002,
            py::arg("puls_flat_threshold") = 0.001,
            py::arg("puls_flat_window_frames") = 10,
            py::arg("puls_median_window_frames") = 200)
        .def("reset", &SignalQualityTracker::reset)
        .def_property_readonly("enabled", &SignalQualityTracker::enabled)
        .def_property_readonly("wants_pulse", &SignalQualityTracker::wants_pulse)
        .def("step_ecg_sample", &SignalQualityTracker::step_ecg_sample)
        .def(
            "build_components",
            [](SignalQualityTracker &tracker,
               const std::vector<SignalQualitySample> &block,
               const py::object &puls_value_norm,
               bool puls_available)
            {
                return tracker.build_components(
                    block,
                    optional_double(puls_value_norm),
                    puls_available);
            },
            py::arg("ecg_quality_block"),
            py::arg("puls_value_norm"),
            py::arg("puls_available"))
        .def(
            "feature_from_components",
            &SignalQualityTracker::feature_from_components);

    py::class_<FeatureFramer>(module, "NativeFeatureFramer")
        .def(py::init([](const py::dict &config)
        {
            return FeatureFramer(feature_config_from_python(config));
        }))
        .def("reset", &FeatureFramer::reset)
        .def(
            "process_sample",
            [](FeatureFramer &framer,
               double ecg1,
               double ecg2,
               const py::object &puls,
               const py::object &sample_idx,
               const py::object &time_ms,
               const py::object &ecg3)
            {
                return framer.process_sample(raw_sample_from_python(
                    ecg1,
                    ecg2,
                    puls,
                    sample_idx,
                    time_ms,
                    ecg3));
            },
            py::arg("ecg1"),
            py::arg("ecg2"),
            py::arg("puls") = py::none(),
            py::arg("sample_idx_400hz") = py::none(),
            py::arg("time_ms") = py::none(),
            py::arg("ecg3") = py::none())
        .def(
            "process_arrays",
            [](FeatureFramer &framer,
               py::array_t<double, py::array::c_style | py::array::forcecast> ecg1,
               py::array_t<double, py::array::c_style | py::array::forcecast> ecg2,
               const py::object &puls,
               const py::object &sample_indices,
               const py::object &times_ms,
               const py::object &ecg3)
            {
                if (ecg1.ndim() != 1 || ecg2.ndim() != 1 ||
                    ecg1.shape(0) != ecg2.shape(0))
                {
                    throw std::invalid_argument(
                        "ecg1/ecg2 must be one-dimensional arrays of equal length.");
                }
                const auto size = static_cast<std::size_t>(ecg1.shape(0));
                const auto ecg3_values = optional_array_values<double>(
                    ecg3,
                    size,
                    "ecg3");
                const auto puls_values = optional_array_values<double>(
                    puls,
                    size,
                    "puls");
                const auto index_values = optional_array_values<std::int64_t>(
                    sample_indices,
                    size,
                    "sample_indices");
                const auto time_values = optional_array_values<double>(
                    times_ms,
                    size,
                    "times_ms");
                const auto ecg1_values = ecg1.unchecked<1>();
                const auto ecg2_values = ecg2.unchecked<1>();
                std::vector<FeatureFrame> frames;
                frames.reserve(size / std::max<std::size_t>(
                    1,
                    framer.config().frame_size_raw));
                for (std::size_t index = 0; index < size; ++index)
                {
                    RawSample sample;
                    sample.ecg1 = ecg1_values(static_cast<py::ssize_t>(index));
                    sample.ecg2 = ecg2_values(static_cast<py::ssize_t>(index));
                    if (ecg3_values.has_value())
                    {
                        sample.ecg3 = (*ecg3_values)[index];
                    }
                    if (puls_values.has_value())
                    {
                        sample.puls = (*puls_values)[index];
                    }
                    if (index_values.has_value())
                    {
                        sample.sample_idx = (*index_values)[index];
                    }
                    if (time_values.has_value())
                    {
                        sample.time_ms = (*time_values)[index];
                    }
                    auto frame = framer.process_sample(sample);
                    if (frame.has_value())
                    {
                        frames.push_back(std::move(*frame));
                    }
                }
                return frames;
            },
            py::arg("ecg1"),
            py::arg("ecg2"),
            py::arg("puls") = py::none(),
            py::arg("sample_indices") = py::none(),
            py::arg("times_ms") = py::none(),
            py::arg("ecg3") = py::none())
        .def_property_readonly("feature_cols", &FeatureFramer::feature_columns)
        .def_property_readonly("frame_size_raw", [](const FeatureFramer &framer)
        {
            return framer.config().frame_size_raw;
        })
        .def_property_readonly("raw_hz", [](const FeatureFramer &framer)
        {
            return framer.config().raw_hz;
        });

    module.def(
        "live_feature_columns",
        [](std::size_t frame_size_raw,
           bool include_puls,
           const std::vector<std::string> &ecg_columns,
           const std::string &ecg_slope_mode,
           const std::string &signal_quality_mode)
        {
            return live_feature_columns(
                frame_size_raw,
                include_puls,
                ecg_columns,
                parse_ecg_slope_mode(ecg_slope_mode),
                parse_signal_quality_mode(signal_quality_mode));
        },
        py::arg("frame_size_raw") = 8,
        py::arg("include_puls") = true,
        py::arg("ecg_columns") = std::vector<std::string>{"ecg1", "ecg2"},
        py::arg("ecg_slope_mode") = "none",
        py::arg("signal_quality_mode") = "none");

    module.def(
        "signal_quality_component_columns",
        [](const std::string &mode, bool include_puls)
        {
            return signal_quality_component_names(
                parse_signal_quality_mode(mode),
                include_puls);
        },
        py::arg("mode"),
        py::arg("include_puls") = true);

    module.def(
        "signal_quality_feature_from_components",
        [](const std::map<std::string, double> &components,
           const py::object &stats,
           double eps)
        {
            return signal_quality_feature_from_components(
                components,
                signal_quality_stats_from_python(stats),
                eps);
        },
        py::arg("components"),
        py::arg("stats") = py::none(),
        py::arg("eps") = 1e-6);

    module.def(
        "signal_quality_features",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
           const std::vector<std::string> &columns,
           const py::object &stats,
           double eps)
        {
            if (values.ndim() != 2 ||
                static_cast<std::size_t>(values.shape(1)) != columns.size())
            {
                throw std::invalid_argument(
                    "values must have shape (rows, len(columns)).");
            }
            const auto calibration = signal_quality_stats_from_python(stats);
            const auto input = values.unchecked<2>();
            py::array_t<float> output(values.shape(0));
            auto result = output.mutable_unchecked<1>();
            for (py::ssize_t row = 0; row < values.shape(0); ++row)
            {
                std::map<std::string, double> components;
                for (py::ssize_t column = 0; column < values.shape(1); ++column)
                {
                    const auto value = input(row, column);
                    if (std::isfinite(value))
                    {
                        components[columns[static_cast<std::size_t>(column)]] = value;
                    }
                }
                result(row) = static_cast<float>(
                    signal_quality_feature_from_components(
                        components,
                        calibration,
                        eps));
            }
            return output;
        },
        py::arg("values"),
        py::arg("columns"),
        py::arg("stats"),
        py::arg("eps") = 1e-6);

    py::class_<TriggerEvent>(module, "NativeTriggerEvent")
        .def_readonly("sid", &TriggerEvent::sid)
        .def_property_readonly("peak_frame_idx_50hz", [](const TriggerEvent &event)
        {
            return event.peak_frame_idx;
        })
        .def_property_readonly("peak_sample_idx_400hz", [](const TriggerEvent &event)
        {
            return event.peak_sample_idx;
        })
        .def_readonly("peak_time_ms", &TriggerEvent::peak_time_ms)
        .def_property_readonly(
            "emitted_on_frame_idx_50hz",
            [](const TriggerEvent &event) { return event.emitted_on_frame_idx; })
        .def_property_readonly(
            "emitted_on_sample_idx_400hz",
            [](const TriggerEvent &event) { return event.emitted_on_sample_idx; })
        .def_readonly("emitted_on_time_ms", &TriggerEvent::emitted_on_time_ms)
        .def_readonly("probability", &TriggerEvent::probability);

    py::class_<StreamingTriggerDetector>(module, "NativeStreamingTriggerDetector")
        .def(
            py::init<double, std::int64_t>(),
            py::arg("threshold"),
            py::arg("holdoff_frames"))
        .def("reset", &StreamingTriggerDetector::reset)
        .def("set_recovery_threshold", &StreamingTriggerDetector::set_recovery_threshold)
        .def("set_thresholds", &StreamingTriggerDetector::set_thresholds)
        .def_property_readonly("last_event_was_recovery", &StreamingTriggerDetector::last_event_was_recovery)
        .def(
            "update",
            [](StreamingTriggerDetector &detector,
               double probability,
               std::int64_t frame_idx,
               std::int64_t sample_idx,
               const py::object &time_ms,
               const py::object &sid,
               const py::object &recovery_probability,
               const py::object &target_frame_idx,
               const py::object &target_sample_idx,
               const py::object &target_time_ms)
            {
                std::optional<TriggerPosition> target;
                if (!target_frame_idx.is_none() || !target_sample_idx.is_none()) {
                    if (target_frame_idx.is_none() || target_sample_idx.is_none())
                        throw std::invalid_argument("Both target indices must be supplied");
                    target = TriggerPosition{target_frame_idx.cast<std::int64_t>(),
                        target_sample_idx.cast<std::int64_t>(), optional_double(target_time_ms)};
                }
                return detector.update(
                    probability,
                    frame_idx,
                    sample_idx,
                    optional_double(time_ms),
                    sid_as_string(sid),
                    optional_double(recovery_probability), target);
            },
            py::arg("probability"),
            py::arg("current_frame_idx_50hz"),
            py::arg("current_sample_idx_400hz"),
            py::arg("current_time_ms"),
            py::arg("sid") = 0,
            py::arg("recovery_probability") = py::none(),
            py::arg("target_frame_idx") = py::none(),
            py::arg("target_sample_idx") = py::none(),
            py::arg("target_time_ms") = py::none())
        .def(
            "flush",
            [](StreamingTriggerDetector &detector,
               std::int64_t frame_idx,
               std::int64_t sample_idx,
               const py::object &time_ms,
               const py::object &sid)
            {
                return detector.flush(
                    frame_idx,
                    sample_idx,
                    optional_double(time_ms),
                    sid_as_string(sid));
            },
            py::arg("current_frame_idx_50hz"),
            py::arg("current_sample_idx_400hz"),
            py::arg("current_time_ms"),
            py::arg("sid") = 0);

    module.def(
        "streaming_trigger_peaks",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> probabilities,
           double threshold,
           std::int64_t holdoff_frames,
           std::int64_t frame_offset)
        {
            if (probabilities.ndim() != 1)
            {
                throw std::invalid_argument(
                    "probabilities must be one-dimensional.");
            }
            const auto values = probabilities.unchecked<1>();
            StreamingTriggerDetector detector(threshold, holdoff_frames);
            std::vector<std::int64_t> peaks;
            for (py::ssize_t index = 0; index < values.shape(0); ++index)
            {
                const auto frame_idx = frame_offset + index;
                auto event = detector.update(
                    static_cast<double>(values(index)),
                    frame_idx,
                    frame_idx,
                    std::nullopt,
                    "0");
                if (event.has_value())
                {
                    peaks.push_back(event->peak_frame_idx);
                }
            }
            if (values.shape(0) > 0)
            {
                const auto final_idx = frame_offset + values.shape(0) - 1;
                auto event = detector.flush(
                    final_idx,
                    final_idx,
                    std::nullopt,
                    "0");
                if (event.has_value())
                {
                    peaks.push_back(event->peak_frame_idx);
                }
            }
            return peaks;
        },
        py::arg("probabilities"),
        py::arg("threshold"),
        py::arg("holdoff_frames") = 0,
        py::arg("frame_offset") = 0);

    module.def(
        "streaming_trigger_peaks_with_recovery",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> probabilities,
           py::array_t<float, py::array::c_style | py::array::forcecast> recovery,
           double threshold, std::optional<double> recovery_threshold,
           std::int64_t holdoff_frames, std::optional<double> emission_threshold)
        {
            if (probabilities.ndim() != 1 || recovery.ndim() != 1 || probabilities.size() != recovery.size())
                throw std::invalid_argument("primary and recovery probabilities must be equal-length vectors");
            const auto p = probabilities.unchecked<1>();
            const auto r = recovery.unchecked<1>();
            StreamingTriggerDetector detector(threshold, holdoff_frames);
            detector.set_recovery_threshold(recovery_threshold);
            detector.set_thresholds(threshold, emission_threshold.value_or(threshold));
            std::vector<std::int64_t> peaks;
            for (py::ssize_t i = 0; i < p.shape(0); ++i)
            {
                auto event = detector.update(p(i), i, i, std::nullopt, "0", r(i));
                if (event) peaks.push_back(event->peak_frame_idx);
            }
            if (p.shape(0))
            {
                auto event = detector.flush(p.shape(0)-1, p.shape(0)-1, std::nullopt);
                if (event) peaks.push_back(event->peak_frame_idx);
            }
            return peaks;
        }, py::arg("probabilities"), py::arg("recovery_probabilities"),
        py::arg("threshold"), py::arg("recovery_threshold"), py::arg("holdoff_frames") = 10,
        py::arg("emission_threshold") = py::none());
}
