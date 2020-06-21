# Problem Statement
## Emergency vs Non-Emergency Vehicle Classification
* Fatalities due to traffic delays of emergency vehicles such as ambulance & fire brigade is a huge problem. In daily life, we often see that emergency vehicles face difficulty in passing through traffic. So differentiating a vehicle into an emergency and non emergency category can be an important component in traffic monitoring as well as self drive car systems as reaching on time to their destination is critical for these services.

* In this problem, you will be working on classifying vehicle images as either belonging to the emergency vehicle or non-emergency vehicle category. For the same, you are provided with the train and the test dataset. Emergency vehicles usually includes police cars, ambulance and fire brigades.

![alt text](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/08/Emgen.jpg)

# Data Description
1. train.zip: contains 2 csvs and 1 folder containing image data
  * train.csv – [‘image_names’, ‘emergency_or_not’] contains the image name and correct class for 1646 (70%) train images
  * images – contains 2352 images for both train and test sets
2. test.csv: [‘image_names’] contains just the image names for the 706 (30%) test images
3. sample_submission.csv: [‘image_names’,’emergency_or_not­’] contains the exact format for a valid submission (1 - For Emergency Vehicle, 0 - For Non Emergency Vehicle)

# Evaluation Metric
* The evaluation metric for this competition is Accuracy.

# Results
* My solution for the problem ended with an accuracy score of 0.9760765550 and place 30 of 466 (top 6%) in the private leaderboard.

# Data
* You can find the data for the competition and leaderboard here: https://datahack.analyticsvidhya.com/contest/janatahack-computer-vision-hackathon/#LeaderBoard
