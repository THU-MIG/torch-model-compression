import numpy as np
from collections import Counter
import copy


def _fill_size(size, total_size):
    # detect the type
    mix = 1
    for item in size:
        if not isinstance(item, int):
            raise RuntimeError("The size " + str(tuple(size)) + " is wrong")
        mix = mix * item
    if -1 not in size:
        if mix != total_size:
            raise RuntimeError("The size " + str(tuple(size)) + " is wrong")
    else:
        neg_counter = np.sum(np.array(size) < 0)
        if neg_counter != 1:
            raise RuntimeError("The size " + str(tuple(size)) + " is wrong")
        neg_index = size.index(-1)
        mix = 1
        for i in range(0, len(size)):
            if i != neg_index:
                mix = mix * size[i]
        if total_size % mix != 0:
            raise RuntimeError("The shape " + str(tuple(size)) + " is wrong")
        size[neg_index] = total_size // mix


def concatenate(seq, dim):
    # check the seq size
    base_mask_shape = list(seq[0].shape)[:]
    del base_mask_shape[dim]
    for i in range(1, len(seq)):
        currnet_shape = list(seq[i].shape)[:]
        del currnet_shape[dim]
        if base_mask_shape != currnet_shape:
            raise RuntimeError("The shape is wrong, the mask can not been cat together")
    narray_shapes = []
    narrays = []
    for i in range(0, len(seq)):
        narray_shapes.append(seq[i]._narray.shape[:])
        narrays.append(seq[i]._narray.copy())
    narray_shapes = np.array(narray_shapes)
    narray_shapes = np.max(narray_shapes, axis=0, keepdims=False)
    for i in range(0, len(narray_shapes)):
        if dim == i or narray_shapes[i] != 1:
            for k in range(0, len(narrays)):
                if seq[k]._narray.shape[i] == 1 and seq[k].shape[i] != 1:
                    narrays[k] = np.repeat(narrays[k], seq[k].shape[i], i)

    narray = np.concatenate(narrays, axis=dim)
    sum_length = 0
    for i in range(0, len(seq)):
        sum_length += seq[i].shape[dim]
    base_mask_shape.insert(dim, sum_length)
    return from_array(narray, base_mask_shape)


def combine_mask(masks):
    # the masks should have the same size
    if len(masks) <= 1:
        raise RuntimeError("At least 2 items")
    base_mask_shape = masks[0].shape[:]
    for i in range(1, len(masks)):
        if base_mask_shape != masks[i].shape:
            raise RuntimeError("The input mask size should be same")
    mask = Mask(base_mask_shape)
    for i in range(0, len(masks)):
        current_mask = masks[i]
        indexs, dims = current_mask.indexs(True)
        mask.set_mask(indexs, dims)
    return mask


def dim_slice(index, dim):
    slice_index = []
    for _ in range(0, dim):
        slice_index.append(slice(None, None, None))
    slice_index.append(index)
    return tuple(slice_index)


def from_array(mask_array, shape):
    mask = Mask(shape)
    mask._narray = mask_array
    mask.compress()
    return mask


def create_mask(shape, indexs, dims):
    mask = Mask(shape)
    mask.set_mask(indexs, dims)
    return mask


class Mask(object):
    def __init__(self, shape):
        self.shape = list(shape)
        array_dims = []
        for _ in range(0, len(self.shape)):
            array_dims.append(1)
        self._narray = np.zeros(tuple(array_dims), dtype=bool)

    # just support the slice and numpy in the list
    def __getitem__(self, indexs):
        numpy = self._narray.copy()
        shape = self.shape[:]
        numpy_shape = list(numpy.shape)
        current_slice = []
        for i in range(0, len(indexs)):
            index = indexs[i]
            current_length = len(range(0, shape[i])[index])
            shape[i] = current_length
            if numpy_shape[i] == 1:
                current_slice.append(slice())
            else:
                current_slice.append(index)
        numpy = numpy[tuple(current_slice)]
        mask = from_array(numpy, shape)
        return mask

    # set the value just support the value and the Mask
    # the indexs just support the slice
    def __setitem__(self, indexs, value):
        numpy_shape = list(self._narray.shape)
        shape = self.shape[:]
        if isinstance(value, bool):
            current_slice = []
            for i in range(0, len(indexs)):
                index = indexs[i]
                current_length = len(range(0, shape[i])[index])
                if shape[i] == current_length:
                    current_slice.append(slice())
                else:
                    if numpy_shape[i] != 1:
                        self._narray = np.repeat(self._narray, shape[i], i)
                    current_slice.append(index)
            self._narray[current_slice] = value
            self.compress()
        if isinstance(value, Mask):
            value_shape = value.shape[:]
            value_narray = value._narray.copy()
            if len(value_narray.shape) != len(numpy_shape):
                raise RuntimeError("The dim length doesn't match")
            current_slice = []
            for i in range(0, len(indexs)):
                index = indexs[i]
                current_length = len(range(0, shape[i])[index])
                if current_length != value_shape[i]:
                    raise RuntimeError("The dim length doesn't match")
                if shape[i] == current_length:
                    current_slice.append(slice())
                else:
                    if numpy_shape[i] != 1:
                        self._narray = np.repeat(self._narray, shape[i], i)
                    current_slice.append(index)
                if self._narray.shape[i] == 1 and value_narray.shape[i] != 1:
                    self._narray = np.repeat(self._narray, shape[i], i)
                if self._narray.shape[i] != 1 and value_narray.shape[i] == 1:
                    value_narray = np.repeat(value_narray, value_shape[i], i)
            self._narray[current_slice] = value_narray
            self.compress()

    def to_array(self, full=True):
        if full is False:
            return self._narray.copy()
        x = self._narray
        shape = x.shape
        for i in range(0, len(self.shape)):
            if self.shape[i] != self._narray.shape[i]:
                x = np.repeat(x, shape[i], i)
        return x

    def boardcast(self, shape):
        ori_shape = self.shape
        i = 0
        return_array = self._narray.copy()
        return_shape = list(self.shape[:])
        while i < len(ori_shape) or i < len(shape):
            if i >= len(ori_shape):
                return_shape.insert(0, shape[len(shape) - i - 1])
                return_array = np.expand_dims(return_array, 0)
                i += 1
                continue
            if i >= len(shape):
                i += 1
                continue
            if i < len(ori_shape) and i < len(shape):
                if ori_shape[len(ori_shape) - i - 1] == shape[len(shape) - i - 1]:
                    i += 1
                    continue
                if ori_shape[len(ori_shape) - i - 1] == 1:
                    ori_shape[len(ori_shape) - i - 1] = shape[len(shape) - i - 1]
                    i += 1
                    continue
                if shape[len(shape) - i - 1] == 1:
                    i += 1
                    continue
                raise RuntimeError(
                    "The shape "
                    + str(shape)
                    + " can not match the shape "
                    + str(ori_shape)
                )
        return from_array(return_array, return_shape)

    # cast the shape into small size
    def shrinkcast(self, shape):
        ori_shape = self.shape[:]
        narray = self._narray.copy()
        if len(shape) > len(ori_shape):
            raise RuntimeError("The shape " + str(shape) + " is too big")
        indexs, dims = self.indexs(return_dims=True, bool_type=True)
        miss_length = len(ori_shape) - len(shape)
        for i in range(0, len(ori_shape)):
            if i < miss_length:
                if i in dims:
                    index = np.arange(0, len(indexs[i]))[np.logical_not(indexs[i])]
                    if len(index) == 0:
                        raise RuntimeError("The mask is totally True")
                    narray = narray[index[0]]
                else:
                    narray = narray[0]
                continue
            if shape[i - miss_length] == ori_shape[i]:
                continue
            if shape[i - miss_length] == 1:
                if i in dims:
                    index = np.arange(0, len(indexs[i]))[not indexs[i]]
                    if len(index) == 0:
                        raise RuntimeError("The mask is totally True")
                    narray = narray[
                        dim_slice(slice(index[0], index[0] + 1), i - miss_length)
                    ]
                else:
                    narray = narray[dim_slice(slice(0, 1), i - miss_length)]
                continue
            raise RuntimeError("The shape " + str(shape) + " is wrong")
        return from_array(narray, shape)

    def compress(self):
        narray_size = list(self._narray.shape)[:]
        for i in range(0, len(narray_size)):
            if narray_size[i] == 1:
                continue
            first_row = self._narray[dim_slice(0, i)]
            compressable = True
            for j in range(1, narray_size[i]):
                current_row = self._narray[dim_slice(j, i)]
                if (first_row == current_row).all() == False:
                    compressable = False
                    break
            if compressable:
                self._narray = self._narray[dim_slice(slice(0, 1), i)]

    def set_mask(self, indexs, dims, value=True):
        indexs = list(filter(lambda item: len(item) != 0, indexs))
        if len(indexs) != len(dims):
            raise RuntimeError("The input params is wrong")
        narray_shape = list(self._narray.shape)[:]
        for i in range(0, len(dims)):
            dim = dims[i]
            if dim < 0:
                dim = dim + len(indexs)
            index = indexs[i]
            if narray_shape[dim] == 1:
                self._narray = np.repeat(self._narray, self.shape[dim], dim)
            self._narray[dim_slice(index, dim)] = value

    def indexs(self, return_dims=False, bool_type=False, force_valid=True):
        narray = np.logical_not(self._narray)
        dims = []
        indexs = []
        if np.sum(narray) == 0:
            raise RuntimeError("All the data is masked")
        narray_shape = list(narray.shape)
        for i in range(0, len(narray_shape)):
            if narray_shape[i] == 1:
                indexs.append([])
                continue
            all_dims = list(range(0, len(narray_shape)))
            del all_dims[i]
            sum_mask = np.sum(narray, axis=tuple(all_dims))
            max_value = max(sum_mask)
            valid_dim = True
            for j in range(0, len(sum_mask)):
                if sum_mask[j] != 0 and sum_mask[j] != max_value:
                    if force_valid:
                        raise RuntimeError("The mask can not extract indexs")
                    else:
                        valid_dim = False
                        break
            if not valid_dim:
                indexs.append([])
                continue
            mask_index = sum_mask == 0
            if np.sum(mask_index) == 0:
                indexs.append([])
                continue
            if bool_type:
                indexs.append(mask_index)
            else:
                indexs.append(np.arange(0, len(sum_mask))[mask_index])
            dims.append(i)
            # check the the other demision
            check_array = narray[dim_slice(mask_index, i)]
            check_array_shape = list(check_array.shape)
            first_row = check_array[dim_slice(0, i)]
            for j in range(1, check_array_shape[i]):
                current_row = check_array[dim_slice(j, i)]
                if (first_row == current_row).all() == False:
                    raise RuntimeError("The mask can not extract indexs")
        if return_dims:
            return indexs, dims
        else:
            return indexs

    def __eq__(self, other):
        if not isinstance(other, Mask):
            return False
        if other.shape != self.shape:
            return False
        if list(other._narray.shape) != list(self._narray.shape):
            return False
        if (other._narray == self._narray).all() == True:
            return True
        return False

    def copy(self):
        shape = self.shape[:]
        narray = self._narray.copy()
        return from_array(narray, shape)

    # divide is simple
    def divide_dim(self, dim, size):
        size = list(size)
        ori_shape = list(self.shape)[:]
        narray = self._narray.copy()
        narray_shape = list(self._narray.shape)[:]
        dim_size = ori_shape[dim]
        _fill_size(size, dim_size)
        # change the shape
        del ori_shape[dim]
        for i in range(dim, dim + len(size)):
            ori_shape.insert(i, size[i - dim])
        # change the narray size
        narray_dim_size = narray_shape[dim]
        del narray_shape[dim]
        for i in range(dim, dim + len(size)):
            if narray_dim_size != 1:
                narray_shape.insert(i, size[i - dim])
            else:
                narray_shape.insert(i, 1)
        narray = np.reshape(narray, narray_shape)
        mask = from_array(narray, ori_shape)
        return mask

    # combine is complex, the 1 and not 1 dim should be considered.
    def combine_dim(self, dims=None):
        if dims is None:
            dims = range(0, len(self.shape))
        if len(dims) < 2:
            return copy.deepcopy(self)
        # check the dims
        for i in range(0, len(dims) - 1):
            if dims[i] + 1 != dims[i + 1]:
                raise RuntimeError("The dims " + str(dims) + " is wrong")
        return_shape = self.shape[:]
        narray = self._narray.copy()
        narray_shape = list(narray.shape)[:]
        begin_dim = dims[0]
        return_shape_dim_size = 1
        narray_shape_dim_size = 1
        for _ in range(0, len(dims)):
            return_shape_dim_size = return_shape_dim_size * return_shape[begin_dim]
            narray_shape_dim_size = narray_shape_dim_size * narray_shape[begin_dim]
            del return_shape[begin_dim]
            del narray_shape[begin_dim]
        return_shape.insert(begin_dim, return_shape_dim_size)
        if narray_shape_dim_size == 1:
            narray_shape.insert(begin_dim, 1)
            narray = np.reshape(narray, narray_shape)
        else:
            ori_shape = self.shape[:]
            ori_narray_shape = list(narray.shape)[:]
            total_size = 1
            for dim in dims:
                total_size = total_size * ori_shape[dim]
                if ori_narray_shape[dim] == 1:
                    narray = np.repeat(narray, ori_shape[dim], dim)
            narray_shape.insert(begin_dim, total_size)
            narray = np.reshape(narray, narray_shape)
        return from_array(narray, return_shape)

    def transpose(self, dims):
        narray = np.transpose(self._narray, dims)
        shape = list(np.array(self.shape)[np.array(dims)])
        return from_array(narray, shape)

    def reduce(self, dims=None, keepdims=False):
        narray = None
        if dims is None:
            narray = np.max(self._narray, keepdims=keepdims)
        else:
            narray = np.max(self._narray, axis=tuple(dims), keepdims=keepdims)
        shape = list(self.shape)[:]
        mask = np.ones(len(shape), dtype=bool)
        mask[np.array(dims)] = False
        if keepdims:
            shape = np.array(shape)
            shape[np.logical_not(mask)] = 1
            shape = list(shape)
        else:
            shape = list(np.array(shape)[mask])
        return from_array(narray, shape)

    def expand_dim(self, dim):
        narray = self._narray.copy()
        shape = self.shape[:]
        shape.insert(dim, 1)
        narray = np.expand_dims(narray, dim)
        return from_array(narray, shape)

    def trim(self, dim):
        indexs, dims = self.indexs(return_dims=True, bool_type=True, force_valid=False)
        narray = self._narray.copy()
        slice_narray = narray
        for current_dim in dims:
            slice_narray = slice_narray[
                dim_slice(np.logical_not(indexs[current_dim]), current_dim)
            ]
        for i in range(0, slice_narray.shape[dim]):
            elements = slice_narray[dim_slice(i, dim)]
            if elements.ndim != 0 and True in slice_narray[dim_slice(i, dim)]:
                slice_narray[dim_slice(i, dim)] = True
            if elements.ndim == 0 and elements == True:
                slice_narray[dim_slice(i, dim)] = True

        return from_array(narray, self.shape[:])

    def include(self, mask):
        if mask.shape != self.shape:
            return False
        indexs, dims = self.indexs(True)
        mask_indexs, mask_dims = mask.indexs(True)
        for mask_dim in mask_dims:
            if mask_dim not in dims:
                return False
            for mask_index in mask_indexs[mask_dim]:
                if mask_index not in indexs[mask_dim]:
                    return False
        return True

    def no_cut(self):
        shape = list(self._narray.shape)
        no_cut = True
        for item in shape:
            if item != 1:
                no_cut = False
                break
        return no_cut

    def slice(self, begin, end, dim, step=None):
        narray = self._narray.copy()
        shape = self.shape[:]
        narray_shape = list(narray.shape)
        if narray_shape[dim] == 1:
            shape[dim] = end - begin
            return from_array(narray, shape)
        else:
            shape[dim] = end - begin
            narray = narray[dim_slice(slice(begin, end, step), dim)]
            return from_array(narray, shape)

    def __str__(self):
        return_string = ""
        return_string += str(self.shape)
        return_string += str(self._narray.shape)
        return return_string

    def __repr__(self):
        return self.__str__()
