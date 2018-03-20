"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import numpy as np
import cv2

from config import Config
import utils
import json
import os

from cv2 import imread


class Dsb2018Dataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """



    def load_dsb2018(self, pathname, is_trainset = True):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        def get_file_data(json_pathname, images_pathname, masks_pathname=None):
            if os.path.exists(json_pathname):
                with open(json_pathname) as data:
                    dictionaries = json.load(data)
            else:
                dictionaries = []

            for dictionary in dictionaries:
                dictionary["image"]["pathname"] = os.path.join(images_pathname, dictionary["image"]["pathname"])
                dictionary["image"]["id"] = dictionary["image"]["pathname"][:-4]
    

                if masks_pathname:
                    for index, instance in enumerate(dictionary["objects"]):
                        dictionary["objects"][index]["mask"]["pathname"] = os.path.join(masks_pathname, dictionary["objects"][index]["mask"]["pathname"])

            return dictionaries


        # Add classes
        self.add_class("dsb2018", 1, "nucleus")

        images_pathname = os.path.join(pathname, "images")
        masks_pathname = os.path.join(pathname, "masks")


        training_pathname = os.path.join(pathname, "training.json")
        training = get_file_data(training_pathname, images_pathname, masks_pathname)
        i=0
        for image in training:
            if i%5==1:
                if is_trainset == False:
                    self.add_image(source='dsb2018',
                       image_id=image['image']['id'],
                       path=image['image']['pathname'],
                       shape=image['image']['shape'], objects = image['objects'])

            else:
                if is_trainset:
                    self.add_image(source='dsb2018',
                       image_id=image['image']['id'],
                       path=image['image']['pathname'],
                       shape=image['image']['shape'], objects = image['objects'])
            i+=1



    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        objects = info['objects']
        instance_masks = []
        class_ids = []
        for obj in objects:
            class_id = 1
            mask = cv2.imread(obj['mask']['pathname'], 0)
            instance_masks.append(mask)
            class_ids.append(self.class_names.index('nucleus'))

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

class Dsb2018DatasetMulticlass(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    class_map = dict()

    def load_dsb2018(self, pathname, class_pathname, is_trainset = True):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        def get_file_data(json_pathname, images_pathname, masks_pathname=None):
            if os.path.exists(json_pathname):
                with open(json_pathname) as data:
                    dictionaries = json.load(data)
            else:
                dictionaries = []

            for dictionary in dictionaries:
                dictionary["image"]["pathname"] = os.path.join(images_pathname, dictionary["image"]["pathname"])
                dictionary["image"]["id"] = dictionary["image"]["pathname"][:-4]


                if masks_pathname:
                    for index, instance in enumerate(dictionary["objects"]):
                        dictionary["objects"][index]["mask"]["pathname"] = os.path.join(masks_pathname, dictionary["objects"][index]["mask"]["pathname"])

            return dictionaries

        def load_image_class(class_pathname):
            import csv
            if os.path.exists(class_pathname):
                with open(class_pathname, ) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['background'] == 'black' and row['foreground'] == 'white':
                            self.class_map[row['filename'][:-4]] = 'wb'
                        elif row['background'] == 'white' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'pw'
                        elif row['background'] == 'white' and row['foreground'] == 'black':
                            self.class_map[row['filename'][:-4]] = 'bw'
                        elif row['background'] == 'yellow' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'py'
                        elif row['background'] == 'purple' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'pp'


        # Add classes
        self.add_class("dsb2018", 1, "wb")
        self.add_class("dsb2018", 2, "py")
        self.add_class("dsb2018", 3, "pw")
        self.add_class("dsb2018", 4, "pp")
        self.add_class("dsb2018", 5, "bw")

        images_pathname = os.path.join(pathname, "images")
        masks_pathname = os.path.join(pathname, "masks")




        training_pathname = os.path.join(pathname, "training.json")
        training = get_file_data(training_pathname, images_pathname, masks_pathname)
        load_image_class(class_pathname)
        i=0
        for image in training:
            if i%10==1:
                if is_trainset == False:
                    self.add_image(source='dsb2018',
                                   image_id=image['image']['id'],
                                   path=image['image']['pathname'],
                                   shape=image['image']['shape'], objects = image['objects'])

            else:
                if is_trainset:
                    self.add_image(source='dsb2018',
                                   image_id=image['image']['id'],
                                   path=image['image']['pathname'],
                                   shape=image['image']['shape'], objects = image['objects'])
            i+=1



    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        objects = info['objects']
        instance_masks = []
        class_ids = []
        for obj in objects:
            mask = cv2.imread(obj['mask']['pathname'], 0)
            instance_masks.append(mask)
            class_name = self.class_map[image_id]
            class_ids.append(self.class_names.index(class_name))

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


class Dsb2018DatasetRam(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    class_map = dict()

    def load_dsb2018(self, pathname, class_pathname, is_trainset = True, clazz = "all"):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # images =

        def get_file_data(json_pathname, images_pathname, masks_pathname=None):
            if os.path.exists(json_pathname):
                with open(json_pathname) as data:
                    dictionaries = json.load(data)
            else:
                dictionaries = []

            for dictionary in dictionaries:
                dictionary["image"]["pathname"] = os.path.join(images_pathname, dictionary["image"]["pathname"])
                dictionary["image"]["id"] = dictionary["image"]["pathname"][:-4]


                if masks_pathname:
                    for index, instance in enumerate(dictionary["objects"]):
                        dictionary["objects"][index]["mask"]["pathname"] = os.path.join(masks_pathname, dictionary["objects"][index]["mask"]["pathname"])

            return dictionaries

        def load_image_class(class_pathname):
            import csv
            if os.path.exists(class_pathname):
                with open(class_pathname, ) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['background'] == 'black' and row['foreground'] == 'white':
                            self.class_map[row['filename'][:-4]] = 'wb'
                        elif row['background'] == 'white' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'pw'
                        elif row['background'] == 'white' and row['foreground'] == 'black':
                            self.class_map[row['filename'][:-4]] = 'bw'
                        elif row['background'] == 'yellow' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'py'
                        elif row['background'] == 'purple' and row['foreground'] == 'purple':
                            self.class_map[row['filename'][:-4]] = 'pp'


        # Add classes
        self.add_class("dsb2018", 1, "wb")
        self.add_class("dsb2018", 2, "py")
        self.add_class("dsb2018", 3, "pw")
        self.add_class("dsb2018", 4, "pp")
        self.add_class("dsb2018", 5, "bw")

        images_pathname = os.path.join(pathname, "images")
        masks_pathname = os.path.join(pathname, "masks")




        training_pathname = os.path.join(pathname, "training.json")
        training = get_file_data(training_pathname, images_pathname, masks_pathname)
        load_image_class(class_pathname)
        i=0
        for image in training:
            if i%10==0:
                if is_trainset == False:
                    self.add_image(source='dsb2018',
                                   image_id=image['image']['id'],
                                   path=image['image']['pathname'],
                                   shape=image['image']['shape'], objects = image['objects'])

            else:
                if is_trainset:
                    self.add_image(source='dsb2018',
                                   image_id=image['image']['id'],
                                   path=image['image']['pathname'],
                                   shape=image['image']['shape'], objects = image['objects'])
            i+=1



    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        objects = info['objects']
        instance_masks = []
        class_ids = []
        for obj in objects:
            mask = cv2.imread(obj['mask']['pathname'], 0)
            instance_masks.append(mask)
            class_name = self.class_map[image_id]
            class_ids.append(self.class_names.index(class_name))

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids