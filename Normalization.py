import numpy as np
import warnings
from sklearn.preprocessing import QuantileTransformer

class Normalizer:
    def __init__(self, data: np.array):
        data = data.astype(np.float32)

        self.raw_data = data.copy()
        self.nan_mask = ~np.isnan(data)
        self.not_nan_raw_data = data[self.nan_mask]

        self.clipped_data = None
        self.not_nan_normalized_data = None
        self.normalized_data = None

    @staticmethod
    def mad_clip(data: np.array, num: float = 5.0) -> np.array:
        if num <= 0.0:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) must be a positive float variable for MAD clip method. 'num' has been adjusted to 0.5.")
            num = 0.5

        if num < 0.5:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) might be too small for MAD clip method.")

        median = np.median(data)
        mad = np.median(np.abs(data - median))

        data_max = median + num * mad
        data_min = median - num * mad

        return np.clip(a=data, a_min=data_min, a_max=data_max)

    @staticmethod
    def sigma_clip(data: np.array, num: float = 3.0) -> np.array:
        if num <= 0.0:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) must be a positive float variable for sigma clip method. 'num' has been adjusted to 0.5.")
            num = 0.5

        if num < 0.5:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) might be too small for sigma clip method.")

        data_mean = np.mean(data)
        data_std = np.std(data)

        data_max = data_mean + num * data_std
        data_min = data_mean - num * data_std

        return np.clip(a=data, a_min=data_min, a_max=data_max)

    @staticmethod
    def quantile_clip(data: np.array, percentile: float = 0.01) -> np.array:
        if percentile <= 0.0:
            warnings.warn(f"[Data Clip]\t Parameter 'percentile' ({round(percentile, 2)}) must be a positive float variable for quantile clip method. 'percentile' has been adjusted to 0.01.")
            percentile = 0.01

        if percentile >= 0.5:
            warnings.warn(f"[Data Clip]\t Parameter 'percentile' ({round(percentile, 2)}) must be a float less than 0.5 for the quantile clip method.")
            percentile = 0.01

        quantile_min = np.quantile(data, percentile)
        quantile_max = np.quantile(data, 1.0 - percentile)

        return np.clip(a=data, a_min=quantile_min, a_max=quantile_max)

    @staticmethod
    def boxing_clip(data: np.array, num: float = 3.0) -> np.array:
        if num <= 0.0:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) must be a positive float variable for boxing clip method. 'num' has been adjusted to 0.5.")
            num = 0.5

        if num < 0.5:
            warnings.warn(f"[Data Clip]\t Parameter 'num' ({round(num, 2)}) might be too small for boxing clip method.")

        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1

        data_max = q3 + num * iqr
        data_min = q1 - num * iqr

        return np.clip(a=data, a_min=data_min, a_max=data_max)

    def zscore_normalize(self) -> None:
        data_mean = np.mean(self.clipped_data)
        data_std = np.std(self.clipped_data)
        if data_std <= 1.0e-15: # avoid all nan values
            warnings.warn(f"[Data Normalization]\t Data must has a non-zero standard deviation. Data std: {data_std}.")
            self.not_nan_normalized_data = np.full_like(self.clipped_data, dtype=np.float32, fill_value=0.0)
        else:
            self.not_nan_normalized_data = (self.clipped_data - data_mean) / data_std

    def _modified_zscore_normalize(self) -> None:
        data_median = np.median(self.clipped_data)
        mad = np.median(np.abs(self.clipped_data - data_median))

        self.not_nan_normalized_data = (self.clipped_data - data_median) / mad

    def _normal_transformer(self, random_state: int = 42) -> None:
        quantile_transformer = QuantileTransformer(n_quantiles=len(self.clipped_data), random_state=random_state, output_distribution='normal')
        clipped_data_reshaped = self.clipped_data.copy().reshape(-1, 1)
        data_transformed = quantile_transformer.fit_transform(clipped_data_reshaped)
        self.not_nan_normalized_data = data_transformed.reshape(-1)

    def _uniform_transformer(self, random_state: int = 42) -> None:
        quantile_transformer = QuantileTransformer(n_quantiles=len(self.clipped_data), random_state=random_state, output_distribution='uniform')
        clipped_data_reshaped = self.clipped_data.copy().reshape(-1, 1)
        data_transformed = quantile_transformer.fit_transform(clipped_data_reshaped)
        self.not_nan_normalized_data = (data_transformed.reshape(-1) - 0.5) * 2

    def _minmax_normalize(self) -> None:
        data_min = np.min(self.clipped_data)
        data_max = np.max(self.clipped_data)

        self.not_nan_normalized_data = (self.clipped_data - data_min) / (data_max - data_min)

    def _l1_normalize(self) -> None:
        abs_sum = np.sum(np.abs(self.clipped_data))
        self.not_nan_normalized_data = self.clipped_data / abs_sum

    def _l2_normalize(self) -> None:
        squared_sum = np.sum(np.square(self.clipped_data))
        self.not_nan_normalized_data = self.clipped_data / np.sqrt(squared_sum)

    def normalize(self, method: str = 'mad-zscore', fill_nan=True) -> np.array:
        method = method.lower()

        # clipping method
        if 'mad' in method:
            self.clipped_data = self.mad_clip(data=self.not_nan_raw_data)
        elif 'sigma' in method:
            self.clipped_data = self.sigma_clip(data=self.not_nan_raw_data)
        elif 'quantile' in method:
            self.clipped_data = self.quantile_clip(data=self.not_nan_raw_data)
        elif 'boxing' in method:
            self.clipped_data = self.boxing_clip(data=self.not_nan_raw_data)
        else:
            self.clipped_data = self.not_nan_raw_data.copy()

        # normalization method
        if 'modified' in method:
            self._modified_zscore_normalize()
        elif 'zscore' in method:
            self.zscore_normalize()
        elif 'normal' in method:
            self._normal_transformer()
        elif 'uniform' in method:
            self._uniform_transformer()
        elif 'minmax' in method:
            self._minmax_normalize()
        elif 'l1' in method:
            self._l1_normalize()
        elif 'l2' in method:
            self._l2_normalize()
        else:
            self.not_nan_normalized_data = self.clipped_data.copy()

        # fill nan value
        if fill_nan:
            self.normalized_data = np.full_like(self.raw_data, fill_value=0.0, dtype=np.float32)
        else:
            self.normalized_data = np.full_like(self.raw_data, fill_value=np.nan, dtype=np.float32)

        self.normalized_data[self.nan_mask] = self.not_nan_normalized_data

        return self.normalized_data
