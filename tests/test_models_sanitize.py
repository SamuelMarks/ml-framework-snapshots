"""Module docstring."""

from ml_framework_snapshots.models import sanitize_type_str


def test_sanitize_type_str() -> None:
    """Function docstring."""
    assert sanitize_type_str(None) is None
    assert sanitize_type_str("<class 'list'>") == "list"
    assert sanitize_type_str("typing.List[int]") == "list[int]"
    assert (
        sanitize_type_str("typing.Dict[str, typing.Tuple[int, int]]")
        == "dict[str, tuple[int, int]]"
    )
    assert sanitize_type_str("Union[list, dict]") == "list | dict"
    assert sanitize_type_str("typing.Union[str, int]") == "str | int"
    assert sanitize_type_str("Optional[int]") == "int | None"
    assert sanitize_type_str("typing.Optional[float]") == "float | None"
    assert sanitize_type_str("typing.Set[int]") == "set[int]"
    assert sanitize_type_str("Type[int]") == "type[int]"
    assert sanitize_type_str("int") == "int"
    assert sanitize_type_str("some_invalid_ast!") == "some_invalid_ast!"
    assert (
        sanitize_type_str("Union[int]") == "Union[int]"
    )  # no transformation if len < 2
    assert sanitize_type_str("typing.Any") == "Any"


def test_sanitize_param_default_edge_cases() -> None:
    """Test sanitize_param_default for non-bracket scrubbed repr and string-only address leaks."""
    from ml_framework_snapshots.models import sanitize_param_default

    class CustomReprWithAngleBrackets:
        """Helper class with angle-bracket repr containing address."""

        def __repr__(self) -> str:
            """Return angle-bracket repr with address.

            Returns:
                String with bracketed address.
            """
            return "<MyObject at 0x7fff1234>"

    class CustomReprWithoutAngleBrackets:
        """Helper class with non-angle-bracket repr containing memory address."""

        def __repr__(self) -> str:
            """Return custom repr with address.

            Returns:
                String with custom address.
            """
            return "custom_object at 0x7fff1234"

    class CustomStrWithAddress:
        """Helper class with clean repr but address in str()."""

        def __repr__(self) -> str:
            """Return clean repr.

            Returns:
                Clean repr string.
            """
            return "CleanRepr"

        def __str__(self) -> str:
            """Return str with address.

            Returns:
                String containing address leak.
            """
            return "address_leak at 0x7fff1234"

    class NormalCustomObject:
        """Helper class with standard custom repr and str."""

        def __repr__(self) -> str:
            """Return standard repr.

            Returns:
                Standard repr string.
            """
            return "NormalCustomObject()"

        def __str__(self) -> str:
            """Return standard str.

            Returns:
                Standard str string.
            """
            return "NormalCustomObject()"

    def_val0, factory0, is_mand0 = sanitize_param_default(CustomReprWithAngleBrackets())
    assert def_val0 == "<factory_default>"
    assert factory0 == "<MyObject>"
    assert is_mand0 is False

    def_val1, factory1, is_mand1 = sanitize_param_default(
        CustomReprWithoutAngleBrackets()
    )
    assert def_val1 == "custom_object"
    assert factory1 is None
    assert is_mand1 is False

    def_val2, factory2, is_mand2 = sanitize_param_default(CustomStrWithAddress())
    assert def_val2 == "<factory_default>"
    assert factory2 == "<factory_default>"
    assert is_mand2 is False

    def_val3, factory3, is_mand3 = sanitize_param_default(NormalCustomObject())
    assert def_val3 == "NormalCustomObject()"
    assert factory3 is None
    assert is_mand3 is False

    # Exception raising value
    class BuggyRepr:
        """Helper class that raises on repr."""

        def __repr__(self) -> str:
            """Raise error on repr.

            Raises:
                RuntimeError: Always raised for testing.
            """
            raise RuntimeError("broken repr")

    def_val4, factory4, is_mand4 = sanitize_param_default(BuggyRepr())
    assert def_val4 == "<unrepresentable>"
    assert factory4 is None
    assert is_mand4 is False


def test_sanitize_param_default_pointer_scrubbing() -> None:
    """Test scrubbing raw hexadecimal pointer addresses from default representations."""
    from ml_framework_snapshots.models import sanitize_param_default

    class PointerDefault:
        """Class with repr containing hex address without at prefix."""

        def __repr__(self) -> str:
            """Return repr with embedded hex address.

            Returns:
                String with hex pointer.
            """
            return "Device(0x7fffbeef)"

    class PointerStrOnly:
        """Class with str containing hex address without at prefix."""

        def __repr__(self) -> str:
            """Return clean repr.

            Returns:
                Clean string.
            """
            return "CleanDevice"

        def __str__(self) -> str:
            """Return str with embedded hex address.

            Returns:
                String with hex pointer.
            """
            return "Device(0x7fffbeef)"

    val_def, fac_def, mand_def = sanitize_param_default(PointerDefault())
    assert val_def == "Device(<addr>)"
    assert fac_def is None
    assert mand_def is False

    val_str, fac_str, mand_str = sanitize_param_default(PointerStrOnly())
    assert val_str == "<factory_default>"
    assert fac_str == "<factory_default>"
    assert mand_str is False


def test_sanitize_framework_constants() -> None:
    """Test preservation of framework dtypes, classes, and device constants."""
    import numpy as np
    from ml_framework_snapshots.models import sanitize_param_default

    # 1. NumPy dtype class
    val_np, fac_np, mand_np = sanitize_param_default(np.float32)
    assert val_np == "numpy.float32"
    assert fac_np is None
    assert mand_np is False

    # 2. Mock TF dtype representation
    class MockTFDtype:
        """Mock TF DType with angle-bracket repr."""

        def __repr__(self) -> str:
            """Return TF dtype representation.

            Returns:
                Dtype string.
            """
            return "<dtype: 'float32'>"

    val_tf, fac_tf, mand_tf = sanitize_param_default(MockTFDtype())
    assert val_tf == "tf.float32"
    assert fac_tf is None
    assert mand_tf is False

    # 3. Mock Torch device representation
    class MockTorchDevice:
        """Mock Torch Device."""

        def __repr__(self) -> str:
            """Return device representation.

            Returns:
                Device string.
            """
            return "device(type='cpu')"

    val_dev, fac_dev, mand_dev = sanitize_param_default(MockTorchDevice())
    assert val_dev == "'cpu'"
    assert fac_dev == "device(type='cpu')"
    assert mand_dev is False
