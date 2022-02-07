import argparse
import logging
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError
from peekingduck.runner import Runner
from peekingduck.pipeline.nodes.input import recorded
from peekingduck.pipeline.nodes.model import yolo
from peekingduck.pipeline.nodes.dabble import fps
from peekingduck.pipeline.nodes.draw import bbox, tag
from peekingduck.pipeline.nodes.output import media_writer, screen
from src.custom_nodes.dabble import person_tracker, passenger_counting
import pdb


log = logging.getLogger(__name__)

def main(args) -> None:
    conf = OmegaConf.load(args.config)
    input_node = recorded.Node(
        input_dir= conf['input.recorded'].input_dir,
        threading=conf['input.recorded'].threading,
        buffer_frame=conf['input.recorded'].buffer_frames
    )
    yolo_node = yolo.Node(
        model_type=conf['model.yolo'].model_type,
        num_classes=conf['model.yolo'].num_classes,
        detect_ids=conf['model.yolo'].detect_ids,
        yolo_iou_threshold=conf['model.yolo'].yolo_iou_threshold,
        yolo_score_threshold=conf['model.yolo'].yolo_score_threshold
    )
    person_tracker_node = person_tracker.Node()
    passenger_counting_node = passenger_counting.Node()
    fps_node = fps.Node()
    draw_node = bbox.Node()
    tag_node = tag.Node()
    if conf.output_to_screen:
        output_node = screen.Node()
    else:
        try:
            assert conf['output.media_writer'].output_dir is not None
            output_node = media_writer.Node(output_dir=conf['output.media_writer'].output_dir)
        except (ConfigAttributeError, ConfigKeyError, AssertionError):
            log.error("If output_to_screen is False, output_dir under output.media_writer must be entered.")
            raise

    runner = Runner(nodes=[
        input_node,
        yolo_node,
        person_tracker_node,
        passenger_counting_node,
        fps_node,
        draw_node,
        tag_node, 
        output_node
    ])

    runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Passenger Counting Algorithm")
    parser.add_argument(
        '--config', 
        type=str, 
        default='./config/passenger_counting/runner_config.yaml',
        help="Path to the config file"
    )
    parsed = parser.parse_args()
    main(args=parsed)