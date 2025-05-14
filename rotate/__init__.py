from .common import rotate_model, RotateOperationRegistry

registry = RotateOperationRegistry()
registry.auto_discover(package_name="model")
