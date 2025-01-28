'''
Read Skin sensor data
==============
**Author**: `zhibin Li`
'''


import argparse
import numpy as np
from vispy import app, scene
from vispy.io import imread, load_data_file, read_mesh
from vispy.scene.visuals import Mesh
from vispy.scene import transforms
from vispy.visuals.filters import TextureFilter
import lib.usb_get as usb_get
import time

class VISPY_3D():
    # 建立模型APP
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                    size=(800, 600))
    view = canvas.central_widget.add_view()


    def __init__(self):
        #初始参数
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--shading', default='smooth',
                            choices=['none', 'flat', 'smooth'],
                            help="shading mode")
        args, _ = self.parser.parse_known_args()

        #读取模型
        self.vertices, self.faces, self.normals, self.texcoords = read_mesh('hand.obj')
        self.texcoords = self.texcoords[:, 0:2]
        self.texture = np.flipud(imread('./UVPNG/skin1024.png'))



        VISPY_3D.view.camera = 'arcball'
        # Adapt the depth to the scale of the mesh to avoid rendering artefacts.
        VISPY_3D.view.camera.depth_value = 10 * (self.vertices.max() - self.vertices.min())
        self.shading = None if args.shading == 'none' else args.shading
        self.mesh = Mesh(self.vertices, self.faces, shading=self.shading, color='white')
        self.mesh.transform = transforms.MatrixTransform()
        self.mesh.transform.rotate(90, (1, 0, 0))
        self.mesh.transform.rotate(135, (0, 0, 1))
        self.mesh.shading_filter.shininess = 1e+1
        VISPY_3D.view.add(self.mesh)

        self.texture_filter = TextureFilter(self.texture, self.texcoords)
        self.mesh.attach(self.texture_filter)

        self.init_start()


        #可视化测试
        # self.texture_flag = 0
        # self.texture2 = np.flipud(imread('./UVPNG/skin5.png'))
        # self.texture_filter2 = TextureFilter(self.texture2, self.texcoords)
        # self.mesh.attach(self.texture_filter2)
        #
        # self.texture1 = np.flipud(imread('./UVPNG/skin4.png'))
        # self.texture_filter1 = TextureFilter(self.texture1, self.texcoords)

        # 初始化USB数据读取，接收
        self.USBDATA_get = usb_get.USB_DataDecode()


    def init_start(self):


        self.attach_headlight(self.mesh, VISPY_3D.view, VISPY_3D.canvas)
        VISPY_3D.canvas.show()
        # 定义计时器
        self.timer = app.Timer(connect=self.update_image, start=True, interval=0.01)

    @canvas.events.key_press.connect
    def on_key_press(self,event):
        if event.key == "t":
            self.texture_filter.enabled = not self.texture_filter.enabled
            self.mesh.update()


    def attach_headlight(self,mesh, view, canvas):
        light_dir = (0, 1, 0, 0)
        mesh.shading_filter.light_dir = light_dir[:3]
        self.initial_light_dir = view.camera.transform.imap(light_dir)

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = VISPY_3D.view.camera.transform
            self.mesh.shading_filter.light_dir = transform.map(self.initial_light_dir)[:3]

    # 清空队列
    def clear_Queue(self, q):
        res = []
        while q.qsize() > 0:
            res.append(q.get(True))


    # 更新函数
    def update_image(self,fram):


        if (self.USBDATA_get.image_construct.empty()==1):
            pass
        else:
            #读取
            self.mesh.detach(self.texture_filter)
            raed_texture = self.USBDATA_get.image_construct.get(True)
            self.texture_filter = TextureFilter(raed_texture, self.texcoords)

            self.mesh.attach(self.texture_filter)
            self.mesh.update()
            self.clear_Queue(self.USBDATA_get.image_construct)
            print(time.time())



if __name__ == "__main__":
    VISPY_3D()
    app.run()