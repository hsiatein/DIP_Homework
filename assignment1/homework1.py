import time
import pyopencl as cl
import numpy as np
from PIL import Image
import gradio as gr

kernel_code = """
__kernel void transform(__global uint* image, __global uint* image2, float scale,
                        float rotation, float trans_x, float trans_y, int rows, int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < 2*rows && j < 2*cols) {
        int x=convert_int_sat((i-trans_x)*cos(rotation)/scale-(j-trans_y)*sin(rotation)/scale);
        int y=convert_int_sat((i-trans_x)*sin(rotation)/scale+(j-trans_y)*cos(rotation)/scale);
        if(0<=x && x<rows && 0<=y && y<cols){
            image2[i * cols *3*2 + j *3 ] = image[x * cols *3 + y *3 ];
            image2[i * cols *3*2 + j *3 +1] = image[x * cols *3 + y *3 +1];
            image2[i * cols *3*2 + j *3 +2] = image[x * cols *3 + y *3 +2];
        }
        else{
            image2[i * cols *3*2 + j *3 ] = 0;
            image2[i * cols *3*2 + j *3 +1] = 0;
            image2[i * cols *3*2 + j *3 +2] = 0;
        }
    }
}
"""

# 初始化 PyOpenCL 环境
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
print(device)
context = cl.Context([device])
queue = cl.CommandQueue(context)
# 编译 OpenCL 程序
program = cl.Program(context, kernel_code).build()


def test(image, scale, rotation, trans_x, trans_y):
    # 将图片转换为 NumPy 数组
    image_array = np.array(image, dtype=np.float32)
    rows, cols, channels = image_array.shape

    time1 = time.time()
    scale = np.float32(scale)
    rotation = np.float32(rotation)
    trans_x = np.float32(trans_x)
    trans_y = np.float32(trans_y)
    white_x2_array = np.full((2 * rows, 2 * cols, 3), 0, dtype=np.float32)
    print(white_x2_array.shape)
    # 创建 OpenCL 缓冲区
    mf = cl.mem_flags
    a_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_array)
    c_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=white_x2_array)
    # 执行内核
    program.transform(queue, white_x2_array.shape, None, a_buffer, c_buffer, scale, rotation
                      , trans_x, trans_y, np.int32(rows), np.int32(cols))
    # 从设备读取结果
    cl.enqueue_copy(queue, white_x2_array, c_buffer).wait()
    # image = Image.fromarray(white_x2_array.astype(np.uint8), 'RGB')
    print(time.time() - time1)
    return white_x2_array.astype(np.uint8)


demo = gr.Interface(
    fn=test,
    inputs=["image", gr.Slider(minimum=0.1, maximum=2, step=0.1), gr.Slider(minimum=0, maximum=3.1, step=0.1),
            gr.Slider(minimum=0, maximum=2000, step=100), gr.Slider(minimum=0, maximum=2000, step=100)],
    outputs=[gr.Image()],
    live=True,
)
demo.launch()
