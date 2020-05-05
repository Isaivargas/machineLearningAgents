import cv2
import numpy as np

# This function process the frames with OpenCv Library resizing the frames to 84x84 frames
def processFrame(frame, shape=(84, 84)):

    """Preprocesses the breakOutGame Env 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8
    # Apply a rgb filter to convert RGB to Gray Scale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # crop image OpenCv2 function to format img[y:y + h, x:x + w]
    frame = frame[34:34+160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    #cv2.imshow('Cropped Image', frame)

    return frame