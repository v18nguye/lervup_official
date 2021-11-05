This directory includes the files needed to predict a rating for a user four real-life situations:

1. class_scores.json - gives the scores of visual concepts in each situation obtained by crowdsourcing. 
 These scores account for the influence of each visual concept in each situation.
 Scores can be negative or positive to reflect their respective influence.
 The larger the absolute value of a score is, the more important its influence will be.
 The names of the visual concepts are anoymized to ensure compliance with GDPR.
 This anonymization has impact in terms of system development since all participants received the same data. 
 The situation acronyms are the following:
 -acc - accommodation search
 -it - job search in IT
 -bank - bank loan search
 -wait - job search as a waiter

2. prediction_train.json, prediction_val.json, prediction_test.json - include the predictions of visual concepts obtained using a Faster-RCNN network trained specifically for the task.
 The file is organized by user and then by images. 
 User and image IDs initially sampled from the YFCC 100M collection but were anonymized in order to ensure GDPR compliance of the dataset.
 This anonymization has no impact in terms of system development since all participants received the same data.
 There are at most 100 images per user but only images with at least one valid detection were included. 
 All predictions whose confidence score is higher or equal to 0.3 are provided for each image.
 The positions of the boxes in the images are also provided. 
 The prediction_test.json data will be provided at a later date.
 
3. gt_train.json, gt_val.json - include the ground truth ratings of users un each of the four situations. gt_test.json will be released after the end of the task.
 These ratings are averaged scores obtained through crowdsourcing.  
 Scores can be negative or positive and reflect the appeal of each user profile in each situation.
 They are useful in order to train systems which predict automatic profile ratings which are as similar as possible to the ground truth ones.
                    
4. dummy_run.json - is a dummy run which includes random automatic ratings for the users included in the validation set.
 It is only useful to illustrate the expected format of the runs submitted by participants.
 Note that automatic rating should be produced independently for each situation since user rating vary from one situation to another.
 They are combined in this file only to facilitate the evaluation of submitted runs.

The images themselves are not provided in order to protect the anonymity of the users included in the dataset.
All participants have access to the data which will allow them to automatically compute user ratings and thus propose a solution to the task.

A more detailed description of the file format is provided below.

### 1.class_scores.json
Dictionary including the scores of visual concepts by situation.

{
    VC_a { 
        'acc' : score_acc,
        'it' : score_it,
        'bank': score_bank,
        'wait': score_wait,
    },
    VC_b {
        ...
}

### 2. Inference results prediction_[train|val|test].json 
Dictionary containing the inference results, with user ids as keys, and user images as inner keys. 
Predictions are stored as an array with the following fields for each element:
 -VC_a - anonymized name of the visual concept. Note that the same visual concept can be detected more than once per image.
 -conf_a - confidence score associated to VC_x
 -x_0, y_0, x_1, y_1 - coordinates of the bounding box associated to the predicted VC_x
 
{
    userId1 {
        userImgId1 [
            "VC_a conf_a x_0 y_0 x_1 y_1",
            "VC_b conf_b x_0 y_0 x_1 y_1",
        ]
        userImgId2 [
            ...
        ]
        ...
    },
    userId2 {
        ...
}

### 3. gt_[train|val].json
Dictionary containing the ground truth ratings for the training and validation sets.

{
    userId1 { 
        'acc' : score_acc,
        'it' : score_it,
        'bank': score_bank,
        'wait': score_wait,
    },
    userId2 {
        ...
}

### 4. dummy_run.json
Dummy run for validation set including randomly produced automatic ratings of of user profiles.
The format is the same as that of the ground truth files.

{
    userId1 { 
        'acc' : score,
        'it' : score,
        'bank': score,
        'wait': score,
    },
    userId2 {
        ...
}
