# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


from tensorflow.python.eager import context


def _executing_eagerly():
    """Returns true if eager execution is supported and enabled."""
    return context.executing_eagerly()


def _make_subgraph(f):
    return tf.function(f)


def _cache(f):
    cache = dict()

    def wrapper(*args):
        key = (args, _executing_eagerly())

        if key in cache:
            return cache[key]
        else:
            retval = f(*args)
            cache[key] = retval
            return retval

    return wrapper


def vars_to_refs(vars):
    if isinstance(vars, list):
        return tuple(vars_to_refs(v) for v in vars)
    return vars.ref()


def refs_to_vars(refs):
    if isinstance(refs, tuple):
        return [refs_to_vars(r) for r in refs]
    return refs.deref()


def _min_allreduced_indexed_slices(values, indices, shape):
    """Reduce allgathered IndexedSlices by values and index

    The operation segments the value rows by unique indices
    and apply the respective min operation

    Arguments:
        values: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`
        indices: A 1-D integer `Tensor` with shape `[D0]`
        shape: a tuple of the dense tensor shape

    Returns:
        A IndexedSlice reduced by the corresponding mapping destination
        of the dense tensor
    """
    unique_indices, new_index_positions = tf.unique(indices)
    new_values = tf.math.unsorted_segment_min(values, new_index_positions, tf.shape(unique_indices)[0])
    return tf.IndexedSlices(new_values, unique_indices, dense_shape=shape)

def _max_allreduced_indexed_slices(values, indices, shape):
    """Reduce allgathered IndexedSlices by values and index

    The operation segments the value rows by unique indices
    and apply the respective min operation

    Arguments:
        values: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`
        indices: A 1-D integer `Tensor` with shape `[D0]`
        shape: a tuple of the dense tensor shape

    Returns:
        A IndexedSlice reduced by the corresponding mapping destination
        of the dense tensor
    """
    unique_indices, new_index_positions = tf.unique(indices)
    new_values = tf.math.unsorted_segment_max(values, new_index_positions, tf.shape(unique_indices)[0])
    return tf.IndexedSlices(new_values, unique_indices, dense_shape=shape)
