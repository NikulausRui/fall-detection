import argparse
import os

# from utils import FeatureExtractor, KeyPoints
from utils import FallDetector

parser = argparse.ArgumentParser(description="Process a video. ")
parser.add_argument(
    "--video", metavar="Video", help="The video to be processed", required=True
)
parser.add_argument(
    "-m",
    "--method",
    metavar="Method",
    help="The type of the cost calculated. Available methods: Division, MeanDifference, DifferenceMean, DifferenceSum, Mean",
    nargs="?",
    default="DifferenceMean",
    const="DifferenceMean",
)
parser.add_argument(
    "--save", action=argparse.BooleanOptionalAction, help="Save or not save the image"
)

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs("assets/outputs", exist_ok=True)
    # featureextractor = FeatureExtractor()
    featureextractor = FallDetector()
    # cost = featureextractor.realTimeVideo(
    cost = featureextractor.process_video(
        video_path=args.video,
        cost_method=args.method,
        save_output=args.save
    )
        # str(args.video), str(args.method), args.save)
