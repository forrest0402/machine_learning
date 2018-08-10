import tensorflow as tf
import numpy as np

_y = np.array([[12, 12, 3], [17, 18, 18], [1, 1, 2]], dtype=np.int32)
hl = np.array([[1], [100], [20]], dtype=np.float32)
hr = np.array([[2], [10], [300]], dtype=np.float32)
x = np.stack([hl, hr], axis=1)
predictions = np.argmin(x, axis=1)
correct_predictions = np.equal(np.reshape(np.equal(_y[:, 1], _y[:, 2]), [-1]), np.reshape(predictions, [-1]))
accuracy = np.mean(correct_predictions)
print("self.accuracy", accuracy)
# with tf.Session() as sess:
#     x = sess.run(ab)
#     print(x)
#     print(x.get_shape())
print("hello, world")
