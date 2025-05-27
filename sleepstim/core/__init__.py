# Make functions from core modules available at the sleepstim.core level

from .io import (
    process_raw_EDF_cfs, 
    read_xml, 
    load_xdf, 
    channel_parser, 
    unravel_hypnogram_visbrain, 
    process_raw_EDF, 
    unravel_hypnogram
)

from .dsp import (
    surface_laplacian, 
    bfr_butter_filt, 
    re_reference, 
    downsample_scaled, 
    bandpower, 
    surface_laplacian_rt
)

from .features import (
    feature_extraction, 
    ecg_feature_extraction, 
    lziv
)

from .classification import (
    sleep_staging, 
    rf_model_select, 
    channel_failure_test
)

from .stimulation import (
    SO_detection, 
    PinkNoise, 
    subject_param_pull
)

from .plotting import (
    plot_confusion_matrix, 
    ROC_curve_plot, 
    plot_multiclass_ROC, 
    transition_matrix_plot
)

from .utils import (
    get_coords, 
    transition_matrix, 
    transition_matrix_prob, 
    thresholdcrossings
)

__all__ = [
    # io
    "process_raw_EDF_cfs", "read_xml", "load_xdf", "channel_parser", 
    "unravel_hypnogram_visbrain", "process_raw_EDF", "unravel_hypnogram",
    # dsp
    "surface_laplacian", "bfr_butter_filt", "re_reference", "downsample_scaled", 
    "bandpower", "surface_laplacian_rt",
    # features
    "feature_extraction", "ecg_feature_extraction", "lziv",
    # classification
    "sleep_staging", "rf_model_select", "channel_failure_test",
    # stimulation
    "SO_detection", "PinkNoise", "subject_param_pull",
    # plotting
    "plot_confusion_matrix", "ROC_curve_plot", "plot_multiclass_ROC", 
    "transition_matrix_plot",
    # utils
    "get_coords", "transition_matrix", "transition_matrix_prob", 
    "thresholdcrossings"
]
