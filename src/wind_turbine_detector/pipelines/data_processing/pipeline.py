from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (extract_relevant_raw_data,
                    label_naip_images,
                    label_overlapping_boxes,
                    remove_overlaped_labels)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_relevant_raw_data,
            inputs="turbines",
            outputs="turbines_gpd",
            name="extract"
        ),
        node(
            func=label_naip_images,
            inputs=["naip_images",
                    "turbines_gpd",
                    "params:distance_ratio"],
            outputs="labeled_dataset",
            name="label"
        ),
        node(
            func=label_overlapping_boxes,
            inputs="labeled_dataset",
            outputs="labeled_dataset_overlaps",
            name="find_overlapped"
        ),
        node(
            func=remove_overlaped_labels,
            inputs="labeled_dataset_overlaps",
            outputs="geodata_with_labels",
            name="remove_overlapped"
        )
    ])
