# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    video.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: taston <thomas.aston@ed.ac.uk>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/25 10:00:46 by taston            #+#    #+#              #
#    Updated: 2023/05/30 10:09:03 by taston           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cv2

class Video:
    """
    A class representing a Video

    ...
    
    Attributes
    ----------
    cap : 

    filename : str
        path to video file
    fps : int
        framerate in frames per second
    height : int
        pixel height of video
    total_frames : int
        length of video in frames
    width : int
        pixel width of video
    writer : 


    Methods
    -------
    create_writer()
        creates video writer object
    get_dim()
        gets video dimensions
    get_fps()
        gets video fps
    get_length()
        gets length of video in frames
    """
    def __init__(self, filename=None):
        if filename:
            self.filename = filename
            self.cap = self._open_vid()
            self.width, self.height = self.get_dim()
            self.total_frames = self.get_length()
            self.fps = self.get_fps()

    def _open_vid(self):
        '''
        Open specified video file and create capture object
        '''
        cap = cv2.VideoCapture(self.filename)
        return cap
    
    def create_writer(self):
        '''
        Create writer object for recording videos based on chosen video.
        '''
        self.writer = cv2.VideoWriter('calibration.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (self.width, self.height))
 
    def get_dim(self):
        '''
        Get video resolution
        '''
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_length(self):
        '''
        Get length of video in frames
        '''
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames
    
    def get_fps(self):
        '''
        Get framerate of video in fps
        '''
        fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        return fps
    
    def __str__(self):
        return ('-'*60 + '\n' + 
                'Video data:' + '\n' + 
                '-'*60 + '\n' + 
                f'Video resolution: {self.width} x {self.height} pixels' + '\n' +
                f'Length of video: {self.total_frames} frames' + '\n' + 
                f'Framerate: {self.fps} fps' + '\n' +
                '-'*60)
        # return 'Video'
    
    

    