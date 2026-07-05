from ultralytics.engine.validator import convert_ndjson_to_yolo_if_needed


def test_convert_ndjson_to_yolo_if_needed_leaves_yaml_unchanged():
    data = "ultralytics/cfg/datasets/VisDrone.yaml"
    assert convert_ndjson_to_yolo_if_needed(data) == data
