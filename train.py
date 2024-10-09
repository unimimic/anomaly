from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
    num_workers=0,
    name="hazelnut_toy",
    root="datasets/hazelnut_toy",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
    task="classification",
)

# Create the datamodule
# datamodule = Folder(
#     name="hazelnut_toy",
#     root="datasets/hazelnut_toy",
#     normal_dir="good",
#     test_split_mode=TestSplitMode.SYNTHETIC,
# )

# Setup the datamodule
datamodule.setup()

# Create the model and engine
model = Patchcore()

# Train a Patchcore model on the given datamodule
engine = Engine(task="classification")
engine.fit(model, datamodule=datamodule)

# Export the model
engine.export(model, export_type=ExportType.OPENVINO, export_root="./export")
