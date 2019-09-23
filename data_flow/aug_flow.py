from albumentations import Flip,HorizontalFlip,ShiftScaleRotate,RandomBrightnessContrast

class aug_tool():

    def __init__(self,images=None,annotations=None,targets=None,shuffle=True,rot_range=0.0,h_flip=False,v_flip=False,shift_range=0,scale=0,luminance=0,contrast=0):

        self.images = images
        self.annotations = annotations
        self.targets = targets
        self.shuffle = shuffle
        self.rot_range = rot_range
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.scale = scale
        self.shift_range = shift_range
        self.luminance = luminance
        self.contrast = contrast

    def get_compose(self):
        augments = []
        if self.h_flip:
            augments.append(HorizontalFlip())

        if self.v_flip:
            augments.append(Flip())

        if (self.luminance >=0 or self.contrast >=0) :
            augments.append(RandomBrightnessContrast(self.luminance,self.contrast))

        if (self.sel.rot_range >= 0 or self.scale >= 0 or self.shift_range >= 0):
            augments.append(ShiftScaleRotate(shift_limit=self.shift_range,rot_range=self.rot_range,scale_limit=self.scale))

        return Compose(augments,p=0.75)

    def get_seg_augments(self,image,mask):
        data = {'image':image,'mask':mask}
        augmented = self.get_compose(**data)
        return augmented['image'],augmented['mask']

    def get_class_augments(self,image):
        data = {'image':image}
        augmented = self.get_compose(image=image)
        return augmented['image']
        



