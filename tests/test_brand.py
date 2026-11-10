import numpy as np
import pytest

from seascape.outputs import brand


@pytest.mark.parametrize("weight", ["Regular", "Medium"])
def test_the_typeface_is_the_brand_books(weight: brand.Weight) -> None:
    assert brand.font(20, weight).getname() == (brand.FAMILY, weight)


def test_the_typeface_has_the_glyphs_a_caption_uses() -> None:
    def drawn(char: str) -> tuple[tuple[int, int], bytes]:
        mask = brand.font(20).getmask(char)
        return mask.size, np.array(mask).tobytes()

    # A glyph the font lacks draws its notdef box, as U+FFFF does.
    assert all(drawn(char) != drawn("\uffff") for char in "°+-0w")
