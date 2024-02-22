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
from nerf.load_llff import viewmatrix, normalize

def projection_intrinsic(intrinsic,n=0.1,f=1000.0):
    f_x = intrinsic[0,0]
    f_y = intrinsic[1,1]
    W_2 = intrinsic[0,2]
    H_2 = intrinsic[1,2]
    return f_x, f_y, W_2, H_2
    # return np.array([[f_x/W_2,    0,              0,              0],
    #                  [  0,    f_y/H_2,            0,              0],
    #                  [  0,      0,     -(f+n)/(f-n), -(2*f*n)/(f-n)],
    #                  [  0,      0,               -1,              0]]).astype(np.float32)

class SLIDEMeshRenderer:

    def __init__(self,
                 height=1024,
                 width=1024,
                 vertices = None,
                 colors=None,
                 alphas=None,
                 indices = None,
                 indices_back = None,
                 img_file="images/moon.png",
                 img_file_back = None,
                 texture_dims=None,
                 frames_path = 'frames'):
        self.texture_dims = texture_dims
        self.program_id = None
        self.vertices = vertices
        self.colors = colors
        self.indices = indices
        self.indices_back = indices_back
        self.img_file = img_file
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
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(self.width, self.height, name, None, None) #(in the accompanying source code this variable will be global)
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

    
    def loadShaders(self,
                    vertex_shader_path="./shaders/renderSLIDEVertexShader.vertexshader",
                    fragment_shader_path="./shaders/renderSLIDEFragmentShader.fragmentshader"):
        self.program_id = common.LoadShaders(vertex_shader_path, fragment_shader_path)

    def initVertices(self):  # We call this right after our OpenGL window is created.
        if self.vertices is None:
            vertices = [
                #  positions
                0.5,  0.5,  1.0, # top right 1
                0.5, -0.5,  1.0, # bottom right 3
                -0.5, -0.5, 1.0, # bottom left 2
                -0.5,  0.5, 1.0, # top left 0
            ]
            self.vertices = np.array(vertices, dtype=np.float32)

        if self.indices is None:
            indices = [
                0, 1, 3, # first triangle
                1, 2, 3  # second triangle
            ]
            self.indices = np.array(indices, dtype=np.uint32)

        if self.colors is None:
            uv = [
                # texture coords
                0.0, 1.0,  # top left
                0.0, 0.0, # bottom left
                1.0, 0.0, # bottom right
                1.0, 1.0 # top right
            ]
            self.colors = np.array(uv, dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Bind the buffer
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        array_type = GLfloat * len(self.vertices)
        glBufferData(GL_ARRAY_BUFFER, len(self.vertices)*4, self.vertices, GL_STATIC_DRAW)
        # glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * 4, array_type(*list(self.vertices)), GL_STATIC_DRAW)


        # Tex Buffer Object
        self.TBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.TBO)
        array_type = GLfloat * len(self.colors)
        glBufferData(GL_ARRAY_BUFFER, len(self.colors) * 4, self.colors, GL_STATIC_DRAW)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TBO)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

    def getProgramLocations(self):
        self.mvp_id = glGetUniformLocation(self.program_id, "MVP")
        # self.tex_prog_id = glGetUniformLocation(self.program_id, "myTextureSampler")

    def getPerspectiveMatrix(self, fov=45.0, ratio=1.0, near=0.1, far=100.0, intrinsics=None):
        if intrinsics is None:
            self.fov = fov
            self.ratio = ratio
            self.zFar = far
            self.zNear = near
            self.P_mat = mat4.perspective(fov, ratio, near, far)

        else:
            f_x, f_y, W, H = projection_intrinsic(intrinsics)
            self.P_mat = mat4.intrinsics_perspective(f_x, f_y, W, H)
            # inchToMm = 25.4
            # self.fov = (360.0/np.pi)*np.arctan((ratio*inchToMm)/intrinsics)
            # self.ratio = ratio
            # self.zFar = far
            # self.zNear = near

    def getViewMatrix(self,
                      camera_pos = [0,0,0.0],
                      look_at = [0,0,0],
                      up = [0,1,0],
                      c2w = None):
        if c2w is None:
            self.V_mat = mat4.lookat(vec3(*camera_pos), vec3(*look_at), vec3(*up))
        else:
            cam_center = (c2w[:3,3].reshape(-1)).tolist()
            # cam_center[-1] = -8
            vec2view = normalize(c2w[:3,2]).reshape(-1).tolist()
            vecup = c2w[:3,1].reshape(-1).tolist()
            # V = viewmatrix(vec2view, vecup, cam_center)
            # c_pos = (c2w[:-1,3]).tolist()
            l_at = mat4.lookat(vec3(*camera_pos), vec3(*look_at), vec3(*up))
            v_mat = mat4(*c2w.reshape(-1).tolist())
            b = mat4.lookat(vec3(*cam_center), vec3(*vec2view), vec3(*vecup))
            # b = np.linalg.inv(V).reshape(-1).tolist()
            # b = np.eye(4).reshape(-1).tolist()
            # b = c2w.reshape(-1).tolist()

            # b = c2w.reshape(-1).tolist()
            self.V_mat = b
            # self.V_mat = l_at*self.V_mat
        

    def getModelMatrix(self):
        self.M_mat = mat4.identity()

    def getMVP(self):
        self.MVP = self.P_mat*self.V_mat*self.M_mat

    def runCircleZoomWindow(self, z_init=1.0, get_screenshot=False, n_frames=1000):
        if not self.window_init():
            return

        # Enable key events
        glfw.set_input_mode(self.window, glfw.STICKY_KEYS,GL_TRUE)

        # Set opengl clear color to something other than red (color used by the fragment shader)
        glClearColor(0.0,0.0,0.0,0.0)

        # Enable depth test
        # glEnable(GL_DEPTH_TEST)
        # # Accept fragment if it closer to the camera than the former one
        # glDepthFunc(GL_LESS)

        glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO)
        glBlendEquation(GL_FUNC_ADD)
        self.loadShaders()
        self.initVertices()
        self.getProgramLocations()
        self.getPerspectiveMatrix(fov=15.0)
        self.getViewMatrix(camera_pos=[0.0, 0.0, -z_init],
                           up=[0,1,0])
        self.getModelMatrix()
        if get_screenshot:
            frame_count = 0
        count = 0.0
        a = np.linspace(-2.0*np.pi,2.0*np.pi,n_frames)
        while glfw.get_key(self.window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(self.window):
            glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.program_id)

            # Send our transformation to the currently bound shader,
            # in the "MVP" uniform
            s, c = np.sin(a[int(count)]), np.cos(a[int(count)])
            if z_init > 0.0:
                self.getViewMatrix(camera_pos=[0.5*s,0.5*c,-z_init + (1/n_frames)*count],
                                   look_at=[0,0,0],
                                   up=[1,0,0])
            else:
                self.getViewMatrix(camera_pos=[0.5*s,0.5*c,z_init + (1/n_frames)*count],
                                   look_at=[0,0,0],
                                   up=[1,0,0])
            # self.V_mat.rotatex(np.pi/2)
            if count > n_frames-2:
                count = 0.0
            else:
                count += 1.0
            if get_screenshot:
                if count > 0.0:
                    frame_count += 1
                else:
                    frame_count = 0.0
                    get_screenshot=False
                    break
            self.getMVP()
            glUniformMatrix4fv(self.mvp_id, 1, GL_FALSE,self.MVP.data)

            # # Bind our texture in Texture Unit 0
            # glActiveTexture(GL_TEXTURE0)
            # glBindTexture(GL_TEXTURE_2D, self.tex_id)
            # # Set our "myTextureSampler" sampler to user Texture Unit 0
            # glUniform1i(self.tex_prog_id, 0)

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
                1,                  # attribute 0. No particular reason for 1, but must match the layout in the shader.
                4,                  # len(vertex_data)
                GL_FLOAT,           # type
                GL_TRUE,           # ormalized?
                0,                  # stride
                self.null                # array buffer offset (c_type == void*)
                )

            # if not self.indices_back is None:
            #     glActiveTexture(GL_TEXTURE0)
            #     glBindTexture(GL_TEXTURE_2D, self.tex_id_back)
            #     # Set our "myTextureSampler" sampler to user Texture Unit 0
            #     glUniform1i(self.tex_prog_id, 0)
            #     # Draw the triangle !
            #     # glDrawArrays(GL_TRIANGLES, 0, len(self.indices)) #3 indices starting at 0 -> 1 triangle
            #     glDrawElements(GL_TRIANGLES, len(self.indices_back), GL_UNSIGNED_INT,  self.indices_back)
            # # Bind our texture in Texture Unit 0
            # glActiveTexture(GL_TEXTURE0)
            # glBindTexture(GL_TEXTURE_2D, self.tex_id)
            # # Set our "myTextureSampler" sampler to user Texture Unit 0
            # glUniform1i(self.tex_prog_id, 0)
            # Draw the triangle !
            # glDrawArrays(GL_TRIANGLES, 0, len(self.indices)) #3 indices starting at 0 -> 1 triangle

            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT,  self.indices)
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
        # glDeleteTextures([self.tex_prog_id])
        glDeleteVertexArrays(1, [self.VAO])

        glfw.terminate()




if __name__ == "__main__":
    randTex = TextureRenderer()
    randTex.runZoomWindow()