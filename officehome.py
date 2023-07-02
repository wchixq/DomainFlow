import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

from utils.torch_img_utils import cv_pad2sq


class OfficeHomeDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, np.asarray(labels)

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        """ each image has been padded to keep aspect ratio """
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHomeDupDataset(OfficeHomeDataset):
    num_class = 65

    def __init__(self, dataset_root, domains, mode='train', transforms=None):
        super().__init__(dataset_root, domains, mode, transforms)

    def __getitem__(self, index):
        image = self.get_original_img(index)
        image1 = None
        if self.transforms is not None:
            image, image1 = self.transforms(image)
        return image, image1

    def __len__(self):
        return len(self.img_list)


class OfficeHomeOriginalResDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, np.asarray(labels)

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return img

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        """ each image has been padded to keep aspect ratio """
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHomeSketchDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_dir, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_dir
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, np.asarray(labels)

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        """ each image has been padded to keep aspect ratio """
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHomeSketchOriginalResDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_dir, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_dir
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, labels

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return img

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        """ each image has been padded to keep aspect ratio """
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHome2TypeDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.struct_dir = dataset_root + '/OfficeHome_structured'
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, labels

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_struct_img(self, index):
        img_file = self.struct_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        img_struct = self.get_struct_img(index)
        """ each image has been padded to keep aspect ratio """
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin, img_struct = self.transforms(img_origin, img_struct)
        return img_origin, img_struct, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHomeReducedDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domain, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domain = domain
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % domain, dtype=str))
        labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % domain, dtype='i8'))
        reduced_ids = np.loadtxt(self.origin_dir + '/reduced1024_%s_ids.txt' % domain, dtype='i8')
        self.labels = np.asarray(labels)[reduced_ids]
        self.img_list = np.asarray(images_list)[reduced_ids]

    def get_square_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_square_img(index)
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHome10ShotDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domain, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domain = domain
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)

        # images_list = np.loadtxt(self.origin_dir + '/%s_images.txt' % domain, dtype=str)
        # labels = np.loadtxt(self.origin_dir + '/%s_labels.txt' % domain, dtype='i8')
        # active_list, active_labels = [], []
        # for n in range(self.num_class):
        #     class_list = images_list[labels == n]
        #     ids = random.sample(population=range(len(class_list)), k=10)
        #     active_list.extend(class_list[ids])
        #     class_labels = np.repeat(n, 10)
        #     active_labels.extend(class_labels)
        # active_list = np.asarray(active_list)
        # active_labels = np.asarray(active_labels)
        # np.savetxt(self.origin_dir + '/10shot_%s_imageslist.txt' % domain, active_list, fmt='%s')
        # np.savetxt(self.origin_dir + '/10shot_%s_labels.txt' % domain, active_labels, fmt='%d')
        # print('10 shot ids for %s generated!' % domain)
        # exit(0)
        self.img_list = np.loadtxt(self.origin_dir + '/10shot_%s_imageslist.txt' % domain, dtype=str)
        self.labels = np.loadtxt(self.origin_dir + '/10shot_%s_labels.txt' % domain, dtype='i8')

    def get_square_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_square_img(index)
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHome20ShotDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domain, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.domain = domain
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)

        num_shot = 20
        # images_list = np.loadtxt(self.origin_dir + '/%s_images.txt' % domain, dtype=str)
        # labels = np.loadtxt(self.origin_dir + '/%s_labels.txt' % domain, dtype='i8')
        # active_list, active_labels = [], []
        # for n in range(self.num_class):
        #     class_list = images_list[labels == n]
        #     if len(class_list) < num_shot:
        #         print('%s domain %s category: only %d images available!' % (domain, self.names[n], len(class_list)))
        #         ids = np.random.randint(low=0, high=len(class_list), size=num_shot)  # maybe repeated
        #     else:
        #         ids = random.sample(population=range(len(class_list)), k=num_shot)  # non-repeated
        #     active_list.extend(class_list[ids])
        #     class_labels = np.repeat(n, num_shot)
        #     active_labels.extend(class_labels)
        # active_list = np.asarray(active_list)
        # active_labels = np.asarray(active_labels)
        # np.savetxt(self.origin_dir + '/%dshot_%s_imageslist.txt' % (num_shot,domain), active_list, fmt='%s')
        # np.savetxt(self.origin_dir + '/%dshot_%s_labels.txt' % (num_shot,domain), active_labels, fmt='%d')
        # print('%d shot ids for %s generated!' % (num_shot,domain))
        # exit(0)
        self.img_list = np.loadtxt(self.origin_dir + '/%dshot_%s_imageslist.txt' % (num_shot,domain), dtype=str)
        self.labels = np.loadtxt(self.origin_dir + '/%dshot_%s_labels.txt' % (num_shot,domain), dtype='i8')

    def get_square_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_square_img(index)
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin = self.transforms(img_origin)
        return img_origin, label, index

    def __len__(self):
        return len(self.img_list)


class OfficeHome2TypeDupDataset(Dataset):
    num_class = 65

    def __init__(self, dataset_root, domains, mode='train', transforms=None):
        super().__init__()
        self.transforms = transforms
        self.origin_dir = dataset_root + '/OfficeHomeDataset_10072016'
        self.struct_dir = dataset_root + '/OfficeHome_structured'
        self.domains = domains
        self.mode = mode
        self.names = np.loadtxt(self.origin_dir + '/categories.txt', dtype=str)
        images_list, labels = [], []
        for working_domain in domains:
            images_list.extend(np.loadtxt(self.origin_dir + '/%s_images.txt' % working_domain, dtype=str))
            labels.extend(np.loadtxt(self.origin_dir + '/%s_labels.txt' % working_domain, dtype='i8'))
        self.img_list, self.labels = images_list, labels

    def get_original_img(self, index):
        img_file = self.origin_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_struct_img(self, index):
        img_file = self.struct_dir + '/' + self.img_list[index]
        img = cv2.imread(img_file)
        return cv_pad2sq(img)

    def get_label(self, index):
        label = self.labels[index]
        return label

    def __getitem__(self, index):
        img_origin = self.get_original_img(index)
        img_struct = self.get_struct_img(index)
        """ each image has been padded to keep aspect ratio """
        img_origin_dup = None
        label = self.get_label(index)
        if self.transforms is not None:
            img_origin, img_origin_dup, img_struct = self.transforms(img_origin, img_struct)
        return img_origin, img_origin_dup, img_struct, label, index

    def __len__(self):
        return len(self.img_list)



