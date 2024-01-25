from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
from glew_wish import *
from csgl import *
from PIL.Image import open as pil_open
from PIL.Image import BICUBIC

import common
import glfw
import sys
import os
import numpy as np
from utilities import *

class ParallaxRenderer:

    def __init__(self,
                 height=1024,
                 width=1024,
                 vertices_fore = None,
                 vertices_back = None,
                 uv_coords_fore=None,
                 uv_coords_back=None,
                 indices_fore = None,
                 indices_back = None,
                 img_file_fore="images/moon.png",
                 img_file_back="images/moon.png",
                 texture_dims=None,
                 frames_path = 'frames'):
        self.texture_dims = texture_dims
        self.program_id = None
        self.vertices_fore = vertices_fore
        self.vertices_back = vertices_back
        self.uv_coords_fore = uv_coords_fore
        self.uv_coords_back = uv_coords_back
        self.indices_fore = indices_fore
        self.indices_back = indices_back
        self.img_file_fore = img_file_fore
        self.img_file_back = img_file_back
        self.frames = frames_path
        self.height = height
        self.width = width
        self.zNear = -10
        self.zFar = 10
        self.angle = 0.0
        # self.shaderDict = {GL_VERTEX_SHADER: vertexShaderString, GL_FRAGMENT_SHADER: fragmentShaderString}
        self.xrot = self.yrot = self.zrot = 0.0
        self.window = None
        self.null = c_void_p(0)

    def window_init(self, name = 'Texture Render'):
        if not glfw.init():
            print("Failed to initialize GLFW\n",file=sys.stderr)
            return False

        # Open Window and create its OpenGL context
        self.window = glfw.create_window(self.width, self.height, name, None, None) #(in the accompanying source code this variable will be global)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if not self.window:
            print("Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n",file=sys.stderr)
            glfw.terminate()
            return False

        # Initialize GLEW
        glfw.make_context_current(self.window)
        glewExperimental = True

        # GLEW is a framework for testing extension availability.  Please see tutorial notes for
        # more information including why can remove this code.
        if glewInit() != GLEW_OK:
            print("Failed to initialize GLEW\n",file=sys.stderr);
            return False
        return True

    def LoadTextures(self,
                     path0=None,
                     path1=None,
                     clamp=False,
                     repeat=False,
                     filter_nearest=False,
                     filter_mipmap=True):
        # global texture
        if path0 is not None:
            self.img_file_fore = path0
        if path1 is not None:
            self.img_file_back = path1
        image0 = pil_open(self.img_file_back)
        image1 = pil_open(self.img_file_fore)
        if self.texture_dims != None:
            image0 = image0.resize(self.texture_dims, resample=BICUBIC)
            image1 = image1.resize(self.texture_dims, resample=BICUBIC)


        # Create Texture 0
        self.im_w = image0.size[0]
        self.im_h = image0.size[1]
        image0 = image0.tobytes("raw", "RGBX", 0, -1)

        self.tex_id_back = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id_back)  # 2d texture (x and y size)
        # img_data = np.array(list(image0.getdata()), np.uint8)
        if clamp and not repeat:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        else:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        if filter_nearest and not filter_mipmap:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, self.im_w, self.im_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image0)
        glGenerateMipmap(GL_TEXTURE_2D)

        # Create Texture 0
        self.im_w = image1.size[0]
        self.im_h = image1.size[1]
        image1 = image1.tobytes("raw", "RGBX", 0, -1)

        self.tex_id_fore = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id_fore)  # 2d texture (x and y size)
        # img_data = np.array(list(image1.getdata()), np.uint8)
        if clamp and not repeat:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        else:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        if filter_nearest and not filter_mipmap:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, self.im_w, self.im_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image1)
        glGenerateMipmap(GL_TEXTURE_2D)

    def loadShaders(self,
                    vertex_shader_path="./shaders/renderParallaxVertexShader.vertexshader",
                    fragment_shader_path="./shaders/renderParallaxFragmentShader.fragmentshader"):
        self.program_id = common.LoadShaders(vertex_shader_path, fragment_shader_path)

    def initVertices(self):  # We call this right after our OpenGL window is created.
        if self.vertices_fore is None:
            vertices_fore = [
                #  positions
                0.5,  0.5,  1.0, # top right 1
                0.5, -0.5,  1.0, # bottom right 3
                -0.5, -0.5, 1.0, # bottom left 2
                -0.5,  0.5, 1.0, # top left 0
            ]
            self.vertices_fore = np.array(vertices_fore, dtype=np.float32)
        if self.vertices_back is None:
            vertices_back = [
                #  positions
                0.5,  0.5,  1.0, # top right 1
                0.5, -0.5,  1.0, # bottom right 3
                -0.5, -0.5, 1.0, # bottom left 2
                -0.5,  0.5, 1.0, # top left 0
            ]
            self.vertices_back = np.array(vertices_back, dtype=np.float32)

        if self.indices_fore is None:
            indices_fore = [
                0, 1, 3, # first triangle
                1, 2, 3  # second triangle
            ]
            self.indices_fore = np.array(indices_fore, dtype=np.uint32)

        if self.indices_fore is None:
            indices_fore = [
                0, 1, 3, # first triangle
                1, 2, 3  # second triangle
            ]
            self.indices = np.array(indices_fore, dtype=np.uint32)

        if self.uv_coords_fore is None:
            uv_fore = [
                # texture coords
                0.0, 1.0,  # top left
                0.0, 0.0, # bottom left
                1.0, 0.0, # bottom right
                1.0, 1.0 # top right
            ]
            self.uv_coords_fore = np.array(uv_fore, dtype=np.float32)

        if self.uv_coords_back is None:
            uv_back = [
                # texture coords
                0.0, 1.0,  # top left
                0.0, 0.0, # bottom left
                1.0, 0.0, # bottom right
                1.0, 1.0 # top right
            ]
            self.uv_coords_back = np.array(uv_back, dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Bind the buffer
        self.VBO_fore = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_fore)
        array_type = GLfloat * len(self.vertices_fore)
        glBufferData(GL_ARRAY_BUFFER, len(self.vertices_fore)*4, self.vertices_fore, GL_STATIC_DRAW)

        self.VBO_back = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO_back)
        array_type = GLfloat * len(self.vertices_back)
        glBufferData(GL_ARRAY_BUFFER, len(self.vertices_back)*4, self.vertices_back, GL_STATIC_DRAW)


        # Tex Buffer Object
        self.TBO_fore = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.TBO_fore)
        array_type = GLfloat * len(self.uv_coords_fore)
        glBufferData(GL_ARRAY_BUFFER, len(self.uv_coords_fore) * 4, self.uv_coords_fore, GL_STATIC_DRAW)

        self.TBO_back = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.TBO_back)
        array_type = GLfloat * len(self.uv_coords_back)
        glBufferData(GL_ARRAY_BUFFER, len(self.uv_coords_back) * 4, self.uv_coords_back, GL_STATIC_DRAW)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TBO)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)



    def getProgramLocations(self):
        self.mvp_id = glGetUniformLocation(self.program_id, "MVP")
        self.tex_prog_id = glGetUniformLocation(self.program_id, "myTextureSampler")

    def getPerspectiveMatrix(self, fov=45.0, ratio=1.0, near=0.1, far=100.0):
        self.fov = fov
        self.ratio = ratio
        self.zFar = far
        self.zNear = near

        self.P_mat = mat4.perspective(fov, ratio, near, far)

    def getViewMatrix(self,
                      camera_pos = [0,0,-1],
                      look_at = [0,0,0],
                      up = [0,1,0]):
        self.V_mat = mat4.lookat(vec3(*camera_pos), vec3(*look_at), vec3(*up))

    def getModelMatrix(self):
        self.M_mat = mat4.identity()

    def getMVP(self):
        self.MVP = self.P_mat*self.V_mat*self.M_mat

    def runZoomWindow(self, z_init=1.0):
        if not self.window_init():
            return

        # Enable key events
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS,GL_TRUE)

        # Set opengl clear color to something other than red (color used by the fragment shader)
        glClearColor(0.0,0.0,0.0,0.0)

        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        # Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS)

        self.loadShaders()
        self.initVertices()
        self.LoadTextures()
        self.getProgramLocations()
        self.getPerspectiveMatrix()
        self.getModelMatrix()
        count = 0.0
        while glfw.get_key(self.window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.program_id)

            # Send our transformation to the currently bound shader,
            # in the "MVP" uniform
            if z_init > 0.0:
                self.getViewMatrix(camera_pos=[0.0, 0.0, -z_init + 0.001*count], up=[0,1,0])
            else:
                self.getViewMatrix(camera_pos=[0.0, 0.0, z_init + 0.001*count], up=[0,1,0])
            if count > 900.0:
                count = 0.0
            else:
                count += 1.0
            self.getMVP()

            glUniformMatrix4fv(self.mvp_id, 1, GL_FALSE,self.MVP.data)

            # Bind our texture in Texture Unit 0
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.tex_id)
            # Set our "myTextureSampler" sampler to user Texture Unit 0
            glUniform1i(self.tex_prog_id, 0)

            #1rst attribute buffer : vertices
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glVertexAttribPointer(
                0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # 2nd attribute buffer : colors
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.TBO)
            glVertexAttribPointer(
                1,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                2,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # Draw the triangle !
            # glDrawArrays(GL_TRIANGLES, 0, len(self.indices)) #3 indices starting at 0 -> 1 triangle
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT,  self.indices)
            # Not strictly necessary because we only have
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)


            # Swap front and back buffers
            glfw.swap_buffers(self.window)

            # Poll for and process events
            glfw.poll_events()

        # note braces around vertex_buffer and vertex_array_id.
        # These 2 functions expect arrays of values
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.TBO])
        glDeleteProgram(self.program_id)
        glDeleteTextures([self.tex_prog_id])
        glDeleteVertexArrays(1, [self.VAO])

        glfw.terminate()

    def runCircleZoomWindow(self, z_init=1.0, get_screenshot=False):
        if not self.window_init():
            return

        # Enable key events
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS,GL_TRUE)

        # Set opengl clear color to something other than red (color used by the fragment shader)
        glClearColor(0.0,0.0,0.0,0.0)

        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        # Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS)

        self.loadShaders()
        self.initVertices()
        self.LoadTextures()
        self.getProgramLocations()
        self.getPerspectiveMatrix(fov=15.0)
        self.getViewMatrix(camera_pos=[0.0, 0.0, -z_init],
                           up=[0,1,0])
        self.getModelMatrix()
        if get_screenshot:
            frame_count = 0
        count = 0.0
        a = np.linspace(-2.0*np.pi,2.0*np.pi,1000)
        while glfw.get_key(self.window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.program_id)

            # Send our transformation to the currently bound shader,
            # in the "MVP" uniform
            self.V_mat.rotatex(2.0*np.pi*(0.1*count))
            s, c = np.sin(a[int(count)]), np.cos(a[int(count)])
            if z_init > 0.0:
                self.getViewMatrix(camera_pos=[0.5*s,0.5*c,-z_init + 0.001*count],
                                   look_at=[0,0,0],
                                   up=[0,1,0])
            else:
                self.getViewMatrix(camera_pos=[0.5*s,0.5*c,z_init + 0.001*count],
                                   look_at=[0,0,0],
                                   up=[0,1,0])
            if count > 900.0:
                count = 0.0
            else:
                count += 1.0
            if get_screenshot:
                if count > 0.0:
                    frame_count += 1
                else:
                    frame_count = 0.0
                    get_screenshot=False
            self.getMVP()

            glUniformMatrix4fv(self.mvp_id, 1, GL_FALSE,self.MVP.data)

            # Bind our texture in Texture Unit 0
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.tex_id_back)
            # Set our "myTextureSampler" sampler to user Texture Unit 0
            glUniform1i(self.tex_prog_id, 0)

            #1rst attribute buffer : vertices
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO_back)
            glVertexAttribPointer(
                0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # 2nd attribute buffer : colors
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.TBO_back)
            glVertexAttribPointer(
                1,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                2,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # Draw the triangle !
            # glDrawArrays(GL_TRIANGLES, 0, len(self.indices)) #3 indices starting at 0 -> 1 triangle
            glDrawElements(GL_TRIANGLES, len(self.indices_back), GL_UNSIGNED_INT,  self.indices_back)



            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.tex_id_fore)
            # Set our "myTextureSampler" sampler to user Texture Unit 0
            glUniform1i(self.tex_prog_id, 0)

            #1rst attribute buffer : vertices
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO_fore)
            glVertexAttribPointer(
                0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # 2nd attribute buffer : colors
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.TBO_fore)
            glVertexAttribPointer(
                1,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
                2,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_FALSE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # Draw the triangle !
            # glDrawArrays(GL_TRIANGLES, 0, len(self.indices)) #3 indices starting at 0 -> 1 triangle
            glDrawElements(GL_TRIANGLES, len(self.indices_fore), GL_UNSIGNED_INT,  self.indices_fore)
            if get_screenshot:
                frame_name = str(frame_count)
                path = os.path.join(self.frames,(4-len(frame_name))*"0"+frame_name+'.png')
                out_img = screenshot(self.width, self.height, file_name_out=path)
            # Not strictly necessary because we only have
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)


            # Swap front and back buffers
            glfw.swap_buffers(self.window)

            # Poll for and process events
            glfw.poll_events()

        # note braces around vertex_buffer and vertex_array_id.
        # These 2 functions expect arrays of values
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.TBO])
        glDeleteProgram(self.program_id)
        glDeleteTextures([self.tex_prog_id])
        glDeleteVertexArrays(1, [self.VAO])

        glfw.terminate()


if __name__ == "__main__":
    randTex = ParallaxRenderer()
    randTex.runZoomWindow()