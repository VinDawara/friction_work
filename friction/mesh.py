import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from math import sin, cos
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
from scipy.io import savemat
import pickle
import copy
import csv
import os

""" mesh class object"""
class mesh:

    # Postprocessor attributes
    latticetype = ''
    solvertype = ''
    dt = np.empty
    folder = ''
    skipsteps = 1

    #  The _init_ method or constructor
    def __init__(self, M,N,a,latticetype, pos, neighbors, angles, normal_stiffness, tangential_stiffness, bottom, top, left, right, disp):
        self.Nx = N
        self.Ny = M
        self.a = a
        self.lattice = latticetype
        self.pos = pos
        self.domain = np.array([np.min(pos[:,0]), np.max(pos[:,0]), np.min(pos[:,1]), np.max(pos[:,1])])
        self.neighbors = neighbors
        self.angles = angles
        self.normal_stiffness = normal_stiffness
        self.tangential_stiffness = tangential_stiffness
        self.bottom = bottom
        self.top = top
        self.left = left
        self.right = right
        self.u = disp
        self.bcs = self.bcs()
        self.lbcs = self.lbcs()

    # displacement boundary condition class
    class bcs:
        ids = []
        comp = []
        parser = []

        def extract_dbcs(self):
            dbcs = []   # displacement bc ids
            pin = []    # pin
            comp = []   # displacement component
            parser = [] # parser function

            # variables to find ids which have both type of displacements
            u_ids = []
            v_ids = []

            for c, ids, p in zip(self.comp, self.ids, self.parser):
                if c == 'u' or c == 'uc':
                    dbcs.append(ids)
                    comp.append(c)
                    parser.append(p)
                    u_ids += ids
                elif c == 'v' or c == 'vc':
                    dbcs.append(ids)
                    comp.append(c)
                    parser.append(p)   
                    v_ids += ids 
                elif c == 'pin':
                    pin += ids  

            uv_ids = np.intersect1d(u_ids,v_ids)        
            return [comp, dbcs, parser, pin, u_ids, v_ids, uv_ids] 

    # load boundary condition class
    class lbcs:
        ids = []
        fx = []
        fy = []
        fun = []

"""Function to generate triangular mesh"""
def meshgenerator(M,N,a=1):
    # lattice type
    latticetype = 'triangle'
       
    # defining mesh properties
    neighbors = [0]*M*N
    angles = [0]*M*N
    normal_stiffness = [0]*M*N
    tangential_stiffness = [0]*M*N
    pos = np.zeros(shape = (M*N,2))
    disp = np.zeros(shape = (M*N,2))

    # default bond stiffness
    Kn = 1
    # node index 
    ID = 0
    # Variables to store the nodes of left and right boundary
    left =[]
    right = []

    # defining inline lambda function
    Lex = lambda i,j: i*N + j

    # Developing nodes for 2D triangular lattice
    for i in range(0,M):
        for j in range(0,N):
            # bottom row
            if i==0:
                # left most node
                if j==0:
                    # 3 bonds connectivity
                    neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j)]
                    angles[ID] = [0, 60, 120]
                    left.append(ID)
                # rightmost node    
                elif j == N-1:
                    # 2 bonds connectivity
                    neighbors[ID] = [Lex(i+1,j), Lex(i,j-1)]
                    angles[ID] = [120, 180]
                    right.append(ID)
                # in-between nodes    
                else:
                    # 4 bonds connectivity
                    neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i,j-1)]
                    angles[ID] = [0,60, 120, 180]

            # top row
            elif i == M - 1:
                # leftmost node
                if j == 0:
                    left.append(ID)
                    if i%2 != 0:
                        # 2 bonds connectivity
                        neighbors[ID] = [Lex(i,j+1), Lex(i-1,j)]
                        angles[ID] = [0, -60]
                    else:
                        # 3 bonds connectivity
                        neighbors[ID] = [Lex(i,j+1), Lex(i-1,j), Lex(i-1,j+1)]
                        angles[ID] = [0, -120, -60]
                elif j == N - 1:
                    right.append(ID)
                    if i%2 != 0:
                        # 3 bonds
                        neighbors[ID] = [Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                        angles[ID] = [180, -120, -60]
                    else:
                        # 2 bonds
                        neighbors[ID] = [Lex(i,j-1), Lex(i-1,j)] 
                        angles[ID] = [180, -120] 
                else:
                    if i%2 != 0:
                        # 4 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                    else:
                        # 4 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i,j-1), Lex(i-1,j), Lex(i-1,j+1)] 

                    angles[ID] = [0, 180, -120, -60]
            else:
                if j == 0:
                    left.append(ID)
                    if i%2 != 0:
                        # 3 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j), Lex(i-1,j)]
                        angles[ID] = [0, 60, -60]
                    else:
                        # 5 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i-1,j), Lex(i-1,j+1)]
                        angles[ID] = [0, 60, 120, -120, -60]
                elif j == N - 1:
                    right.append(ID)
                    if i%2 != 0:
                        # 5 bonds
                        neighbors[ID] = [Lex(i+1,j), Lex(i+1,j-1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                        angles[ID] = [60, 120, 180, -120, -60]
                    else:
                        # 3 bonds
                        neighbors[ID] = [Lex(i+1,j), Lex(i,j-1), Lex(i-1,j)]
                        angles[ID] = [120, 180, -120] 
                else:
                    if i%2 != 0:
                        # 6 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j), Lex(i+1,j-1), Lex(i,j-1), Lex(i-1,j-1), Lex(i-1,j)]
                    else:
                        # 6 bonds
                        neighbors[ID] = [Lex(i,j+1), Lex(i+1,j+1), Lex(i+1,j), Lex(i,j-1), Lex(i-1,j), Lex(i-1,j+1)]

                    angles[ID] = [0, 60, 120, 180, -120, -60]
            
            normal_stiffness[ID] = [Kn]*len(neighbors[ID]) 
            tangential_stiffness[ID] = [0.2*Kn]*len(neighbors[ID])
            if i%2 != 0:
                pos[ID] = np.array([(j*a - 0.5*a), (i*a*0.5*math.sqrt(3))])
            else:
                pos[ID] = np.array([(j*a), (i*a*0.5*math.sqrt(3))])

            ID = ID + 1

    
    total_nodes = ID -1
    bottom = list(range(0,N))
    top = list(range(total_nodes-N+1,total_nodes+1))
    return mesh(M,N,a, latticetype, pos, neighbors, angles, normal_stiffness, tangential_stiffness, bottom, top, left, right, disp)

"""Function to create dictionay object for the mesh class object for saving"""
def create_dictionary_object(mesh):
    # mesh attributes
    dict_obj = {'Nx':mesh.Nx, 'Ny':mesh.Ny, 'a':mesh.a, 'lattice': mesh.lattice, \
                'solvertype':mesh.solvertype, 'dt':mesh.dt, 'folder':mesh.folder,\
                'pos':mesh.pos, 'domain':mesh.domain, 'neighbors':mesh.neighbors, \
                'angles':mesh.angles, 'normal_stiffness':mesh.normal_stiffness, \
                'tangential_stiffness':mesh.tangential_stiffness, 'bottom':mesh.bottom, \
                'top':mesh.top, 'left':mesh.left, 'right':mesh.right, 'disp':mesh.u, \
                'domain':mesh.domain, 'skipsteps':mesh.skipsteps}

    # boundary condition attributes
    dict_obj['bcs.comp'] = mesh.bcs.comp
    dict_obj['bcs.parser'] = mesh.bcs.parser
    dict_obj['bcs.ids'] = mesh.bcs.ids

    dict_obj['lbcs.ids'] = mesh.lbcs.ids
    dict_obj['lbcs.fx'] = mesh.lbcs.fx
    dict_obj['lbcs.fy'] = mesh.lbcs.fy
    dict_obj['lbcs.fun'] = mesh.lbcs.fun
    return dict_obj        

               
"""Function to save the mesh object""" 
def save_mesh(mesh, arg):
    # create a dictionary object
    dict_obj = create_dictionary_object(mesh,arg)
    
    if arg == 'all':
        filename = 'meshobj'
    elif arg == 'bcs':
        filename = 'bcs'    

    try:
        with open(mesh.folder + f'{filename}', 'wb') as f:
            pickle.dump(dict_obj,f)
    except: 
        raise AttributeError("supoorted arguments are \'all\', \'bcs\'")              

"""Function to load mesh"""
def load_mesh(dir, objname = None):
    if objname is None:
        objname = 'meshobj'

    with open(dir + f'/{objname}', 'rb') as f:
        dict_obj = pickle.load(f)

    # create mesh object
    meshobj = mesh(M = dict_obj['Ny'], N = dict_obj['Nx'], a = dict_obj['a'],\
                    latticetype = dict_obj['lattice'], pos = dict_obj['pos'], \
                    neighbors = dict_obj['neighbors'], angles = dict_obj['angles'], \
                    normal_stiffness = dict_obj['normal_stiffness'], \
                    tangential_stiffness = dict_obj['tangential_stiffness'],\
                    bottom = dict_obj['bottom'], top = dict_obj['top'], \
                    left = dict_obj['left'], right = dict_obj['right'], \
                    disp = dict_obj['disp'])    
    
    meshobj.dt = dict_obj['dt']
    meshobj.solvertype = dict_obj['solvertype']
    meshobj.folder = dict_obj['folder']
    meshobj.domain = dict_obj['domain']
    meshobj.skipsteps = dict_obj['skipsteps']

    meshobj.bcs.comp = dict_obj['bcs.comp']
    meshobj.bcs.ids = dict_obj['bcs.ids']
    meshobj.bcs.parser = dict_obj['bcs.parser']  
    meshobj.lbcs.ids = dict_obj['lbcs.ids']
    meshobj.lbcs.fx = dict_obj['lbcs.fx']
    meshobj.lbcs.fy = dict_obj['lbcs.fy']
    meshobj.lbcs.fun= dict_obj['lbcs.fun']

    return meshobj


"""Function to create folders"""
def create_directory(folder):
    path = os.getcwd()
    folder = os.path.join(path,folder)
    try:
        os.mkdir(folder)
    except:
        print(f'Directory with name {folder} already exist')  
    return folder    

"""Function to create sub-directory"""
def sub_directory(dir_name, path):
    dir = os.path.join(path, f'{dir_name}/') 
    if not os.path.isdir(dir):
        os.mkdir(dir) 
    return dir
        
"""function to view the mesh lattice"""
def meshplot(mesh, filename = 'latticeview', title = 'latticeview', vectorfield = 'off', save = False, dir = 'snapshots'):
    
    # Creating empty graph
    G = nx.Graph()
    Edgelist = []

    for id in range(0,len(mesh.pos)):
        G.add_node(id, pos = mesh.pos[id])
        for neigh, ns, aph in zip(mesh.neighbors[id], mesh.normal_stiffness[id], mesh.angles[id]):
            if ns:
                if aph in [0, 60, 120]:
                    Edgelist.append([id, neigh])
                    
    G.add_edges_from(Edgelist)
    pos = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize = (10,10), dpi  = 300)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim([mesh.domain[0]-2, mesh.domain[1]+2])
    ax.set_ylim([mesh.domain[2], mesh.domain[3]+2])
    g = nx.draw_networkx_edges(G, pos, width = 0.8)
    if vectorfield == 'on':
        ax.quiver(mesh.pos[:,0], mesh.pos[:,1], mesh.u[:,0], mesh.u[:,1], scale = 20, color = 'blue')
    if save:   
        if not mesh.folder:
            mesh.folder = create_directory(f"{mesh.Ny}X{mesh.Nx}")

        # create snapshots directory insider folder
        snapshots = sub_directory(f'{dir}', mesh.folder)
        
        fig.savefig(snapshots + filename, bbox_inches = 'tight', dpi = 300)
        plt.close()    
    else:    
        plt.show()

"""function to add material properties"""
def bondstiffness(node, kn = 1, R = 0.2, a = 1, nondim = False, **kwargs):
    if kwargs:
        try:
            nu = kwargs['poisson_ratio']
            if nondim == True:
                cl = math.sqrt((1-nu)/((1+nu)*(1-2*nu)))
                cs = math.sqrt(1/(2*(1+nu)))
                cr = cs*(0.874 + 0.162*nu)
                kn = (cl/cr)**2  - (1/3)*(cs/cr)**2
                R = ((cs/cr)**2 - (1/3)*(cl/cr)**2)/kn
        except:
            print('KeywordError: poisson_ratio is required')    

        else:
            if nondim == False:
                try:
                    E = kwargs['youngs_modulus']
                    rho = kwargs['density']
                    cl = math.sqrt(E*(1-nu)/(rho*(1+nu)*(1-2*nu)))
                    cs = math.sqrt(E/(2*rho*(1+nu)))
                    kn = (cl**2  - (1/3)*cs**2)/a**2
                    R = (cs**2 - (1/3)*cl**2)/(kn*a**2)
                except:
                    print('KeywordError: youngs_modulus and density are both needed')    
    
    for id in range(0,len(node.pos)):
        node.normal_stiffness[id] = [kn]*len(node.neighbors[id])
        node.tangential_stiffness[id] = [R*kn]*len(node.neighbors[id])
            
"""Boundary conditions"""

"""function to obtain Dirichlet displacement boundary condition"""
def dispBC(mesh, dbc, arg, value = 0):
    # update mesh object
    parser = f"{value}"
    mesh.bcs.ids.append(dbc)
    mesh.bcs.parser.append(parser)
    mesh.bcs.comp.append(arg)

    if arg == 'u':
        mesh.u[dbc,0] = value
    elif arg == 'v':
        mesh.u[dbc,1] = value
    elif arg == 'pin':
        mesh.u[dbc] = np.zeros(shape=(1,2))        
    

"""Function to obtain Dirichlet function boundary condition"""
def DirichletFunction(mesh,dbc, arg, parser):
    # update the mesh object
    mesh.bcs.ids.append(dbc)
    mesh.bcs.parser.append(parser)
    mesh.bcs.comp.append(arg)
   

"""Parser function for loading"""  
def LoadParserFunction(mesh, ids, fx = '0', fy = '0', fun = 'parser'):
    mesh.lbcs.ids.append(ids)
    if type(fx) != str:
        fx = str(fx)
    if type(fy) != str:
        fy = str(fy)    

    mesh.lbcs.fx.append(fx)
    mesh.lbcs.fy.append(fy)
    mesh.lbcs.fun.append(fun)
       
