# Pri-matrix competititon
## Problem description

Your model should identify the animals (or lack thereof) in a given Chimp&See video. There are 24 categories in total: 23 animal categories plus 1 category corresponding to no animal. Each video is identified by a 10 character alphanumeric string followed by a .mp4, e.g., abcde12345.mp4. This index is referred to as the video's filename. Given a video file and it's filename as input, your trained model should output a list of 24 probabilities corresponding to the model's confidence that each respective category is present in the video.

We have used the crowd-sourced annotations from Chimp&See to generate ground truth labels for each video in the dataset. Some videos have no animals in them, in which case the blank category of the video's labels will be 1 and all other columns will be 0. Otherwise, if a species is present its entry will be a 1. Multiple species may be present!

Wisdom of the masses: a note on crowdsourcing the truth. We have taken many steps to go from raw annotations to a well-labeled dataset. This includes enforcing certain thresholds on how many user annotations are required to accept a label as well as thresholds related to percentages of user agreement. That said, this technique for leveraging crowdsourced data is uncharted territory and there is bound to be some noise!

