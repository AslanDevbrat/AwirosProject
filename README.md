# Action Recognition
- Developed a Starting app that can detect only 3 action.
- The App is based on Pose Landmarks Extraction.
- Created a Dataset of 3 action i.e. “Hello”, “Thanks”, “I Love you”.
- Each action contains the 30 video stored as NumPy arrays
- The array contain the Coordinates of the all the Landmarks, especially Hands Landmarks
- I trained a Sequential LSTM model 

## Dataset
### Action : Hello


https://user-images.githubusercontent.com/39759685/135710215-b8634545-d8f3-4cfb-bd69-465c28d8c137.mp4


### Action : Thank you


https://user-images.githubusercontent.com/39759685/135710222-4d5786e3-0b64-4d48-b752-24a4f130a202.mp4



### Action : I Love YOU

https://user-images.githubusercontent.com/39759685/135709750-412a2033-4a9a-4f6c-bbcc-985139d4e3f8.mp4
## Final App Inference



https://user-images.githubusercontent.com/39759685/135709719-46df639d-1121-4707-ac73-13b953c716ed.mp4

## Model Performance (Confusion Matrixs)
| Action : Hello | Action : Thanks | Action : I love You |
| ----------------------------------------- | --------------------------------- | ------------------------------|   
| ![image](https://user-images.githubusercontent.com/39759685/135710358-61d63aca-5885-403a-9bf3-ed9e66712d34.png) | ![image](https://user-images.githubusercontent.com/39759685/135710376-38a0205b-4176-4ddb-892d-24857e0931fb.png) | ![image](https://user-images.githubusercontent.com/39759685/135710384-0579dbbf-7810-4d36-9aaa-2c3f59e76eca.png) |

