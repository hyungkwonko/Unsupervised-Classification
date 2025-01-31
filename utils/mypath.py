"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'samples', 'thumbnail', 'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return 'data/cifar-10/'
        
        elif database == 'cifar-20':
            return 'data/cifar-20/'

        elif database == 'stl-10':
            return 'data/stl-10/'

        elif database == 'thumbnail':  # for thumbnail
            return 'data/thumbnail/'

        elif database == 'samples':  # for samples
            return 'data/samples/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return 'data/imagenet/'
        
        else:
            raise NotImplementedError
