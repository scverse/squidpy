from typing import Dict, Tuple, Union, Optional
import pytest


class TestIO:
    """TODO."""

    def test_read_metadata_too_large_image(self):
        pass

    @pytest.mark.parametrize(
        "shape",
        [
            (101,),
            (101, 64),
            (1, 101, 64),
            (3, 101, 64),
            (1, 101, 64, 1),
            (1, 101, 64, 3),
            (3, 101, 64, 1),
            (3, 101, 64, 3),
            (3, 101, 64, 3, 3),
        ],
    )
    def test_get_shape(self, shape: Tuple[int, ...]):
        pass

    @pytest.mark.parametrize("shape", [(101,), (101, 64), (1, 101, 64), (3, 101, 64, 1)])
    def test_determine_dimensions(self, shape: Tuple[int, ...]):
        pass

    @pytest.mark.parametrize("chunks", [100, (100, 100), "auto", None, {"y": 100, "x": 100}])
    def test_lazy_load_image(self, chunks: Optional[Union[int, Tuple[int, ...], str, Dict[str, int]]]):
        pass
