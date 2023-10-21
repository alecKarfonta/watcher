import json
import logging
import cv2
from time import time

from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
import torch
from PIL import Image
import requests
import PIL
import numpy as np


class Watcher():


    def __init__(
            self,
            logger:logging.Logger=None, 
            config_filename:str="watcher_config.json", 
            pre_init:bool=False
        ):
        # If provided a logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            #self.logger.setLevel("debug")
        # Log init
        self.logger.debug(f"{__class__.__name__}.init()")

        # Load config
        try:
            self.config = json.load(open(config_filename, "r"))
        except FileNotFoundError:
            self.logger.error(f"{__class__.__name__}.init(): FileNotFoundError Could not read config file at {config_filename}")
            return None
        
        self.model = None
        self.image_processor = None
        # If should pre initialize model
        if pre_init:
            self.init_model()



    def init_model(self, model_name:str=None) -> bool:
        self.logger.debug(f"{__class__.__name__}.init_model()")
        if not model_name:
            model_name = self.config["model"]["name"]
        device = self.config["model"]["device"]

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        #self.model = AutoModelForObjectDetection.from_pretrained(model_name, torch_dtype=torch.bfloat16 )
        self.model.to("cuda")

        return True



    def get_objects(
            self, 
            image:np.ndarray, 
            is_profile:bool=False
        ):
        if is_profile:
            start_time = time()
        self.logger.debug(f"{__class__.__name__}.get_objects()")
        if not self.model:
            self.init_model()

        # Conform image object
        shape = None

        #if isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        #    shape = image.size[::-1]
        if isinstance(image, np.ndarray):
            # Convert to PIL image
            #image = Image.fromarray(image)
            shape = image.shape[:2]
        self.logger.debug(f"{__class__.__name__}.Received image of type {type(image)}")

        #if isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        #    self.logger.debug(f"{__class__.__name__}.get_objects(): Reading a PIL.JpegImagePlugin.JpegImageFile image")
        #else:
        #    self.logger.error(f"{__class__.__name__}.get_objects(): Unrecognized image type: {type(image) = }")
        #    #return None

        # Convert image to model inputs
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs.to("cuda")
        # Without updating the model
        with torch.no_grad():
            # Run inference on the image
            outputs = self.model(**inputs)

        # Convert output
        target_sizes = torch.tensor([shape])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        # Save all objects fond
        objects = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            #box = [round(i, 2) for i in box.tolist()]
            box = [int(round(i, 2)) for i in box.tolist()]
            #self.logger.debug(f"{__class__.__name__}.get_objects()")(
            #    f"Detected {self.model.config.id2label[label.item()]} with confidence "
            #    f"{round(score.item(), 3)} at location {box}"
            #)

            objects.append({
                "label" : self.model.config.id2label[label.item()],
                "score" : float(score),
                "x1" : box[0],
                "y1" : box[1],
                "x2" : box[2],
                "y2" : box[3],
            })

        if is_profile:
            # Measure runtime
            end_time = time()
            run_time = end_time - start_time
            
            model_size = get_model_size(self.model)
            #get_model_size(self.summarizer)
            self.logger.info(f'{__class__.__name__}.get_text_summary(): Executed in {(run_time):.4f}s. Used {model_size} of memory.') 
        
        return objects
    

    def overlay_boxes(self, objects:list, image):

        """
        objects : list[dict] : 
                        {'label': 'chair',
                        'score': 0.9984951019287109,
                        'x1': 237,
                        'y1': 520,
                        'x2': 380,
                        'y2': 847}
        """

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = .75
        fontColor              = (255,0,0)
        thickness              = 2
        lineType               = 2

        height = len(image[0])
        width = len(image)

        for index in range(len(objects)):
            obj = objects[index]
            label = obj["label"]

            # Overlay bounding box
            start_point = (obj["x1"], obj["y1"])
            end_point = (obj["x2"], obj["y2"])
            #print (f"{start_point} : {end_point}")
            cv2.rectangle(image, start_point, end_point, color=fontColor, thickness=2)
            
            # Overlay class label
            loc = (obj["x1"], obj["y1"]) # Start at upper left corner of box
            # If is not close to the bottom
            if loc[1] < height - 10:
                # Lower the y value
                loc = (loc[0], loc[1] - 10)

            # If too close to the top
            if loc[1] < 20:
                # Fix close to the top
                loc = (loc[0], 20)


            cv2.putText(
                image,
                label, 
                loc, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType
            )
            

        return image



def get_model_size(model, in_gb:bool=True) -> str:
    param_size:int = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size:int = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    if in_gb:
        return f"{round((param_size + buffer_size) / 1024**3, 1)} GB"
    else:
        return f"{int((param_size + buffer_size) / 1024**2)} MB"