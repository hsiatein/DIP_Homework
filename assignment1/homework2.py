import time
import cv2
import numpy as np
import gradio as gr
import pyopencl as cl

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img


# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点

    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射

    return marked_image


# 执行仿射变换
kernel_code = """
__kernel void transform(__global float* img,
                                    __global float* wimg,
                                    __global float* param1,
                                    __global float* param2,
                                    __global float2* paramq,
                                    int s0, int s1, int n, float r
                                    ) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < s1 && j < s0) {
        float x = i;
        float y = j;

        for (int k = 0; k < n; k++) {
            x = x + param1[k] * exp(-((i - paramq[k].x) * (i - paramq[k].x) + (j - paramq[k].y) * (j - paramq[k].y)) / (r * r));
            y = y + param2[k] * exp(-((i - paramq[k].x) * (i - paramq[k].x) + (j - paramq[k].y) * (j - paramq[k].y)) / (r * r));
        }

        int x_int = convert_int_sat(x);
        int y_int = convert_int_sat(y);

        if (x_int >= 0 && x_int < s1 && y_int >= 0 && y_int < s0) {
            int img_index = (y_int * s1 + x_int) * 3;
            int wimg_index = (j * s1 + i) * 3;

            wimg[wimg_index] = img[img_index];
            wimg[wimg_index + 1] = img[img_index + 1];
            wimg[wimg_index + 2] = img[img_index + 2];
        }
        else{
            int img_index = (y_int * s1 + x_int) * 3;
            int wimg_index = (j * s1 + i) * 3;
            wimg[wimg_index] = 255;
            wimg[wimg_index + 1] = 255;
            wimg[wimg_index + 2] = 255;
        }
    }
}
"""
# 初始化 PyOpenCL 环境
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context, device)
# 编译 OpenCL 程序
program = cl.Program(context, kernel_code).build()
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    time1=time.time()
    r=image.shape[0]/10
    p=np.array(points_src).astype(np.float32)
    q=np.array(points_dst).astype(np.float32)
    n=len(q)
    D=p-q
    d1=D[:,0]
    d2=D[:,1]
    # print(d1,d2)
    B=np.zeros((n,n)).astype(np.float32)
    for i in range(n):
        for j in range(n):
            B[i,j]=np.exp(-np.linalg.norm(q[i]-q[j])**2/r**2)
    a1=np.linalg.lstsq(B, d1)[0].astype(np.float32)
    a2=np.linalg.lstsq(B, d2)[0].astype(np.float32)
    # print(q)
    # print(a1)
    # print(a2)
    # print(time.time() - time1, "秒")
    # print(image)
    image=np.array(image,dtype=np.float32)
    warped_image = np.zeros_like(image).astype(np.float32)
    mf = cl.mem_flags
    image_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    warped_image_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=warped_image)
    a1_buffer=cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1)
    a2_buffer=cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2)
    q_buffer=cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    #warped_image1=kernel(image,warped_image,a1,a2,q,image.shape[0],image.shape[1],n,r)
    # 执行内核
    program.transform(queue, (np.int32(image.shape[1]),np.int32(image.shape[0]),3), None,image_buffer,warped_image_buffer,a1_buffer,a2_buffer,
                      q_buffer,np.int32(image.shape[0]),np.int32(image.shape[1]),np.int32(n),np.float32(r))
    # 从设备读取结果
    warped_image1=np.full(warped_image.shape,0,dtype=np.float32)
    cl.enqueue_copy(queue, warped_image1, warped_image_buffer).wait()
    print(time.time()-time1,"秒")
    # print(warped_image1)
    return np.array(warped_image1,dtype=np.uint8)


def run_warping():
    global points_src, points_dst, image  ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image( label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", width=800, height=800)

        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)

    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

# 启动 Gradio 应用
demo.launch()
