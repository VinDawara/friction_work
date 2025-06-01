import friction.mesh as fm
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import math
import copy
import pickle
import csv
import h5py


"""Class objects"""
class contact:
    model = ''
    surfaces = []
    contacts = []
    angles = []
    bondlength = []
    param = {}  

fm.mesh.contact = contact()

"""Function to create dictionary object for a mesh class object"""
def create_dict_obj(mesh):
    
    dict_obj = fm.create_dictionary_object(mesh)

    dict_obj['contact.model'] = mesh.contact.model 
    dict_obj['contact.surfaces'] = mesh.contact.surfaces    
    dict_obj['contact.contacts'] = mesh.contact.contacts
    dict_obj['contact.angles'] = mesh.contact.angles
    dict_obj['contact.bondlength'] = mesh.contact.bondlength
    dict_obj['contact.param'] = mesh.contact.param

    return dict_obj

"""Function to save the mesh object"""
def save_mesh(mesh, objname = None):
    # create dictionary  object
    dict_obj = create_dict_obj(mesh)

    if objname is None:
        objname = 'meshobj'   
      
    try:
        with open(mesh.folder + f'/{objname}', 'wb') as f:
            pickle.dump(dict_obj,f)
    except: 
        raise AttributeError("supoorted arguments are \'all\', \'bcs\'") 

"""Function to load mesh object"""   
def load_mesh(dir,objname = None):
    # load the dictionary file
    if objname is None:
        objname = 'meshobj'

    with open(dir + f'/{objname}', 'rb') as f:
        dict_obj = pickle.load(f)

    # first load the mesh object
    meshobj = fm.load_mesh(dir, objname)

    # add contact class from the dictionary
    meshobj.contact.model = dict_obj['contact.model']
    meshobj.contact.surfaces = dict_obj['contact.surfaces'] 
    meshobj.contact.contacts = dict_obj['contact.contacts'] 
    meshobj.contact.angles = dict_obj['contact.angles'] 
    meshobj.contact.bondlength = dict_obj['contact.bondlength'] 
    meshobj.contact.param = dict_obj['contact.param']

    return meshobj
    
"""Function to combine two mesh and create single mesh with interface"""
def interface(mesh1, mesh2, mesh2_xpos = None, thick = None):

    # thickness of the interface
    if thick is None:
        thick = 0.5*np.sqrt(3)*mesh1.a
    mesh1.contact.param['do'] = thick    

    # merging nodes
    total_nodes_1 = mesh1.Nx*mesh1.Ny
    total_nodes_2 = mesh2.Nx*mesh2.Ny
    for id in range(total_nodes_2):
        neighbors = []
        for neigh in mesh2.neighbors[id]:
            neighbors.append(neigh + total_nodes_1)
        mesh2.neighbors[id] = neighbors    

    # aligning mesh2 node postions
    if mesh2_xpos is None:
        shift = mesh1.pos[0] + (0,mesh1.pos[-1][1] + thick)
    else:
        shift = mesh1.pos[0] + (mesh2_xpos, mesh1.pos[-1][1] + thick)

    mesh2.pos = mesh2.pos + shift     
    
    # concatenation of positions and displacement
    mesh1.pos = np.concatenate((mesh1.pos, mesh2.pos))
    mesh1.u = np.concatenate((mesh1.u, mesh2.u))
    
    # domain information
    mesh2.domain = np.array([np.min(mesh2.pos[:,0]), np.max(mesh2.pos[:,0]), np.min(mesh2.pos[:,1]), np.max(mesh2.pos[:,1])])
    mesh1.domain = np.vstack((mesh1.domain, mesh2.domain))

    # concatenation of neighbors list
    mesh1.neighbors += mesh2.neighbors

    # concatenation of stiffnesses
    mesh1.normal_stiffness += mesh2.normal_stiffness
    mesh1.tangential_stiffness += mesh2.tangential_stiffness
    
    # concatenation of angles
    mesh1.angles += mesh2.angles

    # boundary information
    # bottom
    temp = []
    temp.append(mesh1.bottom)
    newids = []
    for id in mesh2.bottom:
        newids.append(id+total_nodes_1)
    mesh2.bottom = newids
    temp.append(mesh2.bottom)
    mesh1.bottom = temp
    # top
    temp = []
    temp.append(mesh1.top)
    newids = []
    for id in mesh2.top:
        newids.append(id+total_nodes_1)
    mesh2.top = newids
    temp.append(mesh2.top)
    mesh1.top = temp
    # left
    temp = []
    temp.append(mesh1.left)
    newids = []
    for id in mesh2.left:
        newids.append(id+total_nodes_1)
    mesh2.left = newids
    temp.append(mesh2.left)
    mesh1.left = temp
    # right
    temp = []
    temp.append(mesh1.right)
    newids = []
    for id in mesh2.right:
        newids.append(id+total_nodes_1)
    mesh2.right = newids
    temp.append(mesh2.right)
    mesh1.right = temp 

    # lattice size information 
    mesh1.Nx = [mesh1.Nx, mesh2.Nx]
    mesh1.Ny = [mesh1.Ny, mesh2.Ny]
    
    # contact information
    mesh1.contact.surfaces.append(mesh1.top[0])
    mesh1.contact.surfaces.append(mesh1.bottom[1])
    contacts, angles, bondlength = nearest_nodes(mesh1)
    mesh1.contact.contacts = contacts
    mesh1.contact.angles = angles    
    mesh1.contact.bondlength = bondlength
    return mesh1       

"""Function to find nearest nodes for the interface nodes"""
def nearest_nodes(mesh):
    surf1 = mesh.contact.surfaces[0]
    surf2 = mesh.contact.surfaces[1]
    actual_contacts = mesh.contact.contacts
    thick = mesh.contact.param['do']
    interfacebonds = []
    angles = []     # angles w.r.t surf 1 nodes
    bondlength = []
    # take an node at surface 1
    for id1 in surf1:
        # search node at surface 2 which is nearest
        for id2 in surf2:
             if [id1, id2] not in actual_contacts:
                r = mesh.pos[id2] - mesh.pos[id1]
                aph = np.arctan2(r[1],r[0])*180/np.pi
                l = np.linalg.norm(r,2)
                ly = r[1]
                if l <= thick+0.05 and ly <= thick:
                    interfacebonds.append([id1, id2]) 
                    angles.append(aph)
                    bondlength.append(l)
    return interfacebonds, angles, bondlength             

"""Function to assign interface bond properties"""
def interface_bond_properties(mesh, model='linear', **kwargs):
    mesh.contact.model = model
    try:
        mesh.contact.param['kn'] = kwargs['kn']
    except:
        mesh.contact.param['kn'] = 0.3
        print('Interface spring stiffness is not defined. Switched to default value')    
    try:
        mesh.contact.param['db'] = kwargs['db']
    except:
        mesh.contact.param['db'] = 1
        print('Interface breaking strength is not defined. Switched to default value')    
    try:
        mesh.contact.param['a'] = kwargs['a']
    except:
        mesh.contact.param['a'] = 1
        print('Bond forming length is not specified. Switched to default value')

    if model == 'double':
        try:
            mesh.contact.param['kt'] = kwargs['kt']
        except:
            raise Exception('Interface tangential stiffness \'kt\' is required for double model')  
    if model == 'double2':
        try:
            mesh.contact.param['kt'] = kwargs['kt']
        except:
            raise Exception('Interface tangential stiffness \'kt\' is required for double2 model')      

              
"""Function to check the state of connected bonds"""
def breaking_condition(mesh):
    # extract interface properties
    interfacebonds = mesh.contact.contacts
    db = mesh.contact.param['db']
   
    total_bonds = len(interfacebonds)
    # length of the bonds
    l = np.zeros(total_bonds)
    for i, [id, neigh] in enumerate(interfacebonds):
        rf = mesh.pos[neigh] - mesh.pos[id]    
        lf = np.linalg.norm(rf,2) 
        l[i] = lf
        
    # check for breaking condition
    brokenbonds = [ids for ids,val in zip(interfacebonds,l) if val>= db]
    return brokenbonds

"""Function to form new bonds"""
def forming_condition(mesh, force, velocity):
    surf1 = mesh.contact.surfaces[0]
    surf2 = mesh.contact.surfaces[1]
    curr_contacts = mesh.contact.contacts
    a = mesh.contact.param['a']
    do = mesh.contact.param['do']

    # extract top and bottom interface bonds ids
    curr_bottom_ids = []
    curr_top_ids =[]
    for id1, id2 in curr_contacts:
        curr_bottom_ids.append(id1)
        curr_top_ids.append(id2)

    newbonds = []
    angles = []     # angles w.r.t surf 1 nodes
    bondlength = []
    # take an node at surface 1 (bottom lattice)
    for id1 in surf1:
        if id1 not in curr_bottom_ids:
            for id2 in surf2:   # (top lattice)
                if id2 not in curr_top_ids:
                    r = mesh.pos[id2] - mesh.pos[id1]
                    aph = np.arctan2(r[1],r[0])*180/np.pi
                    l = np.linalg.norm(r,2)
                    # ly = r[1]
                    # if l <= a and ly <= do:
                    if l <= a:    
                        # cond = np.abs(np.array([force[id1,1], velocity[id1,1], force[id2,1], velocity[id2,1]]))
                        # if np.all(cond <= 1e-5):
                        newbonds.append([id1, id2]) 
                        angles.append(aph)
                        bondlength.append(l)              
    return newbonds, angles, bondlength

"""Function to update contacts"""
def update_contacts(mesh, brokenbonds, newbonds, new_aph, newlen):
    contacts = mesh.contact.contacts
    angles = mesh.contact.angles
    bondlength = mesh.contact.bondlength

    # first remove broken bonds from the contacts
    temp = []
    aph = []
    length = []
    for i, [id1, id2] in enumerate(contacts):
        if [id1, id2] not in brokenbonds:
            temp.append([id1, id2])
            aph.append(angles[i])
            length.append(bondlength[i])
    
    # add new bonds
    contacts = temp + newbonds       
    angles = aph + new_aph
    bondlength = length + newlen

    mesh.contact.contacts = contacts
    mesh.contact.angles = angles
    mesh.contact.bondlength = bondlength
    return contacts

"""function to view the mesh lattice"""
def meshplot(mesh, filename = 'latticeview', title = 'latticeview', vectorfield = 'off', save = False, dir = 'snapshots'):
    path = r'c:\\Users\Admin\Documents\\VScode\\scripts\\friction'
    plt.style.use(path + './article_preprint.mplstyle')
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

    pos = mesh.pos
    contacts = mesh.contact.contacts
    segs = np.zeros(shape = (len(contacts), 2,2))
    # create a line segments
    for i, [id1, id2] in enumerate(contacts):
        x = (pos[id1,0], pos[id2,0])
        y = ([pos[id1,1], pos[id2,1]])
        segs[i,:,0] = x
        segs[i,:,1] = y

    line_collection = LineCollection(segs, linewidths = 1  , color = 'tab:red')  

    fig, ax = plt.subplots(dpi=600)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')
    ax.set_title(r'$%s$'%title)
    ax.set_xlim([mesh.domain[0,0]-2, mesh.domain[0,1]+2])
    ax.set_ylim([mesh.domain[0,2], mesh.domain[1,3]+2]) #0,+2
    g = nx.draw_networkx_edges(G, pos, width = 0.3, edge_color = 'black')
    ax.add_collection(line_collection)
    if vectorfield == 'on':
        ax.quiver(mesh.pos[:,0], mesh.pos[:,1], mesh.u[:,0], mesh.u[:,1], scale = 20, color = 'blue')
    if save:   
        if not mesh.folder:
            mesh.folder = fm.create_directory(f"friction_{mesh.Ny[0]}X{mesh.Nx[0]}")

        # create snapshots directory insider folder
        snapshots = fm.sub_directory(f'{dir}', mesh.folder) 

        fig.savefig(snapshots + filename, bbox_inches = 'tight', dpi = 600)
        plt.close()    
    else:    
        plt.show()

"""Generate A matrix for verlet computation"""
def generate_A_matrix(mesh):
  
    # create the copy of the node object
    neighbors = copy.copy(mesh.neighbors)
    angles = copy.copy(mesh.angles)
    normal_stiff = copy.copy(mesh.normal_stiffness)
    tang_stiff = copy.copy(mesh.tangential_stiffness)

    # declare variables to store A and b
    total_nodes = len(mesh.pos)
    I,J, value = ([] for i in range(3))

    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1
    
    # coefficeints u_i and v_i
    coeff_ux = lambda ns,ts,aph: np.round_(np.array(ns)*np.cos(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.sin(np.pi*np.array(aph)/180)**2, decimals = 12)
    coeff_uv = lambda ns,ts,aph: np.round_((np.array(ns)-np.array(ts))*np.multiply(np.sin(np.pi*np.array(aph)/180),np.cos(np.pi*np.array(aph)/180)), decimals = 12)
    coeff_vy = lambda ns, ts, aph: np.round_(np.array(ns)*np.sin(np.pi*np.array(aph)/180)**2 + np.array(ts)*np.cos(np.pi*np.array(aph)/180)**2, decimals = 12)

    for id in range(0,total_nodes):
        
        # coefficient of u_i in X equilibrium equation
        I.append(u(id)); J.append(u(id))
        value.append(np.sum(coeff_ux(normal_stiff[id], tang_stiff[id], angles[id])))    

        # coefficient of v_i in X equilibrium equation
        I.append(u(id)); J.append(v(id))
        value.append(np.sum(coeff_uv(normal_stiff[id], tang_stiff[id], angles[id])))

        # coefficient of u_i in Y equilibrium equation
        I.append(v(id)); J.append(u(id))
        value.append(np.sum(coeff_uv(normal_stiff[id], tang_stiff[id], angles[id])))

        # coefficient of v_i in Y equilibrium equation
        I.append(v(id)); J.append(v(id))
        value.append(np.sum(coeff_vy(normal_stiff[id], tang_stiff[id], angles[id])))

        # coefficient of u_j and v_j in X and Y force equation
        for neigh, aph, ns, ts in zip(neighbors[id],angles[id],normal_stiff[id],tang_stiff[id]):
            # coefficient of u_j in X equation
            I.append(u(id)); J.append(u(neigh))
            value.append(-coeff_ux(ns,ts,aph))

            # coefficient of v_j in X equation
            I.append(u(id)); J.append(v(neigh))
            value.append(-coeff_uv(ns,ts,aph))

            # coefficient of u_j in Y equation
            I.append(v(id)); J.append(u(neigh))
            value.append(-coeff_uv(ns,ts,aph))

            # ceofficient of v_j in Y equation
            I.append(v(id)); J.append(v(neigh))
            value.append(-coeff_vy(ns,ts,aph))

    value = np.array(value)
    I = np.array(I)
    J = np.array(J)
    
    # Creating a sparse matrix
    A = csc_matrix((value, (I,J)), shape=(2*total_nodes,2*total_nodes))

    return A


"""Generate B matrix for verlet computation"""
def generate_B_matrix(mesh):
  
    # create the copy of the node object
    angles = copy.copy(mesh.angles)
    normal_stiff = copy.copy(mesh.normal_stiffness)


    # declare variables to store b
    total_nodes = len(mesh.pos)
    b = np.zeros(shape=(2*total_nodes,1))

    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1
    
    # coefficeints u_i and v_i
    coeff_constX = lambda ns, aph: np.array(ns)*np.cos(np.pi*np.array(aph)/180)
    coeff_constY = lambda ns, aph: np.array(ns)*np.sin(np.pi*np.array(aph)/180)
    
    for id in range(0,total_nodes):
        b[u(id)] = np.sum(coeff_constX(normal_stiff[id], angles[id]))
        b[v(id)] = np.sum(coeff_constY(normal_stiff[id], angles[id]))

    return b


"""Function to compute interface bonds"""
def interface_force(mesh):
    # read the interface bond information
    interfacebonds = mesh.contact.contacts
    length = mesh.contact.bondlength
    angles = mesh.contact.angles
    model = mesh.contact.model
    # read interface parameters
    ns = mesh.contact.param['kn']
    a = mesh.contact.param['a']
    if model == 'double':    
        ts = mesh.contact.param['kt']
        angles = mesh.contact.angles  
    elif model == 'double2':
        ts = mesh.contact.param['kt']
        angles = mesh.contact.angles
     

    total_nodes = len(mesh.pos)
    fb = np.zeros(shape=(total_nodes,2))

    for i, [id,neigh] in enumerate(interfacebonds):
        rf = mesh.pos[neigh] - mesh.pos[id]
        lf = np.linalg.norm(rf,2)
        if model == 'linear':
            fb[id] = ns*(lf-length[i])*(rf/lf)   
        elif model == 'double':
            ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
            dr = np.dot((rf-ri),rf/lf)*rf/lf
            fn = ns*dr
            ft = ts*((rf-ri) - dr)   
            fb[id] = fn + ft 
        elif model == 'double2':
            ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
            fn = ns*(rf[1]-a)
            ft = ts*(rf[0]-ri[0])
            fb[id] = np.array([ft,fn])
            
        fb[neigh] = -fb[id]

    fb = flatten(fb)
    return fb

"""Function to compute linearized interface force bonds"""
def linear_interface_force(mesh):
    # read the interface bond information
    interfacebonds = mesh.contact.contacts
    length = mesh.contact.bondlength
    angles = mesh.contact.angles
    model = mesh.contact.model
    # read interface parameters
    ns = mesh.contact.param['kn']
    a = mesh.contact.param['a']
    if model == 'double':    
        ts = mesh.contact.param['kt']
        angles = mesh.contact.angles  
    elif model == 'double2':
        ts = mesh.contact.param['kt']
        angles = mesh.contact.angles      
        
    total_nodes = len(mesh.pos)
    fb = np.zeros(shape=(total_nodes,2))

    for i, [id,neigh] in enumerate(interfacebonds):
        aph = angles[i]
        ro = np.array([np.cos(aph*np.pi/180), np.sin(aph*np.pi/180)])
        rf = mesh.pos[neigh] - mesh.pos[id]
        lf = np.linalg.norm(rf,2)
        if model == 'linear':
            fb[id] = ns*(lf-length[i])*ro 
        elif model == 'double':
            ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
            dr = np.dot((rf-ri),rf/lf)*rf/lf
            fn = ns*dr
            ft = ts*((rf-ri) - dr)   
            fb[id] = fn + ft 
        elif model == 'double2':
            ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
            fn = ns*(rf[1]-a)
            ft = ts*(rf[0]-ri[0])
            fb[id] = np.array([ft,fn])       

        fb[neigh] = -fb[id]

    fb = flatten(fb)
    return fb

"""impose boundary conditons for verlet integrator"""
def dispbcs(disp,bcs_comp, bcs_ids, bcs_parser,pin, t):  
    
    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1
     
    for nbc in range(len(bcs_comp)):
        id = np.array(bcs_ids[nbc])
        if bcs_comp[nbc] == 'u' or bcs_comp[nbc] == 'uc':
            disp[u(id)] = eval(bcs_parser[nbc])
                
        elif bcs_comp[nbc] == 'v' or bcs_comp[nbc] == 'vc':
            disp[v(id)] = eval(bcs_parser[nbc])
            
    if pin:
        id = np.array(pin)
        disp[u(id)] = 0
        disp[v(id)] = 0    
                
    return disp                     

"""impose complementary boundary conditons for verlet integrator"""
def dispbcs_complementary(initial_disp, bcs_comp, bcs_ids):  
    
    # variables suffix corresponding to node id 'i'
    u = lambda i: 2*i
    v = lambda i: 2*i + 1

    disp = np.zeros(shape = (len(initial_disp),1)) 
    for nbc in range(len(bcs_comp)):
        id = np.array(bcs_ids[nbc])
        if bcs_comp[nbc] == 'uc':
            disp[u(id)] = initial_disp[u(id)]
                
        elif bcs_comp[nbc] == 'vc':
            disp[v(id)] = initial_disp[v(id)]               
    return disp 

"""Function to impose load boundary conditions for verlet integrator"""
def loadbcs(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t):
    # define lambda for getting indices
    u = lambda i: 2*i
    v = lambda i: 2*i + 1

    load = np.zeros(shape=(2*total_nodes,1))

    for i in range(len(lbcs_fun)):
        id = np.array(lbcs_ids[i])
        if lbcs_fun[i] == 'parser':
            load[u(id)] = eval(lbcs_fx[i])
            load[v(id)] = eval(lbcs_fy[i])
        elif lbcs_fun[i] == 'impulse' and t == 0:
            load[u(id)] = eval(lbcs_fx[i])
            load[v(id)] = eval(lbcs_fy[i])
        elif lbcs_fun[i] == 'ramp':
            if t<=10: 
                load[u(id)] = (t/10)*eval(lbcs_fx[i])
                load[v(id)] = (t/10)*eval(lbcs_fy[i])
            else:
                load[u(id)] = eval(lbcs_fx[i])
                load[v(id)] = eval(lbcs_fy[i])
    return load            


"""reshape displacement to array(total_nodes,2) to make it callable with other functions""" 
def reshape2vector(u):
    # variables suffix corresponding to node id 'i'
    u_idx = lambda i: 2*i
    v_idx = lambda i: 2*i + 1
    n = int(0.5*len(u))
    u_reshape = np.zeros(shape = (n,2))
    for id in range(n):
        u_reshape[id,0] = u[u_idx(id)]
        u_reshape[id,1] = u[v_idx(id)]
    return u_reshape 

"""reshape displacement from array(total_nodes,2) to array(2*total_nodes,1)""" 
def flatten(u):
    # variables suffix corresponding to node id 'i'
    u_idx = lambda i: 2*i
    v_idx = lambda i: 2*i + 1
    n = len(u)
    u_reshape = np.zeros(shape = (2*n,1))
    for id in range(n):
        u_reshape[u_idx(id)] = u[id,0]
        u_reshape[v_idx(id)] = u[id,1]
    return u_reshape 

"""Function to flatten the list"""
def list_flatten(list):
    temp = []
    for item in list:
        temp += item
    return temp    

# function to save mesh object in csv file for visualization
def write_to_csv(field,  t, filename = 'LatticeInfo'):
    filename = f"{filename}.csv"
    field = [t, field]
    with open(filename,'a') as csvfile:
        # create a csv write object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(field) 

"""Function to implement dynamic integration"""
def dynamic_solver(mesh, dt, endtime, c = 0, vectorfield = 'off', folder = 'd_sol', **kwargs):
    # checking the keywords
    if kwargs:
        interval = kwargs['interval']
    else:
        interval = False

    # simulation parameters
    mesh.dt = dt  
    # creating directory for storing data
    if not mesh.folder:
        mesh.folder = fm.create_directory(f"{folder}_{mesh.Ny[0]}X{mesh.Nx[0]}")
    # extract positions of nodes  
    pos = copy.deepcopy(mesh.pos)

    # extract displacement boundary conditions
    bcs_comp, bcs_ids, bcs_parser,pin,_,_,_ = mesh.bcs.extract_dbcs()

    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh.lbcs.ids)
    lbcs_fx = copy.copy(mesh.lbcs.fx)
    lbcs_fy = copy.copy(mesh.lbcs.fy)
    lbcs_fun = copy.copy(mesh.lbcs.fun)

    # total number of iterations
    maxsteps = math.ceil(endtime/dt)

    # variables to store the displacement field for postprocessing
    total_nodes = len(pos)

    # create A and B matrix without interface bonds
    A = generate_A_matrix(mesh)
    B = generate_B_matrix(mesh)
    # interface force
    mesh.pos = pos + mesh.u
    fb = linear_interface_force(mesh)
    
    # displacement variables for time step 't-1', 't', and 't+1', respectively
    # for verlet algorithm
    u_prev, u_curr, u_next = (flatten(mesh.u) for i in range(3))

    # impose boundary condition at t = 0
    u_curr = dispbcs(u_curr,bcs_comp, bcs_ids, bcs_parser, pin, t = 0)
    ro = flatten(pos)
    r_curr = ro +  u_curr
    u_prev = copy.deepcopy(u_curr)
    
    # time step for convergence check
    conv_time = endtime/10
    # integration begins
    for step in range(maxsteps):
        # time variable
        t = step*dt

        # velocity of the node
        v = (u_next - u_prev)/(2*dt)

        # impose load boudnary conditions
        load = loadbcs(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t)
        # time integration
        u_next = 2*u_curr - u_prev - (A@r_curr + B - fb - load + c*v)*dt**2

        # impose boundary conditions
        u_next = dispbcs(u_next, bcs_comp, bcs_ids, bcs_parser, pin, t)

        # interface force
        mesh.u = reshape2vector(u_next)
        mesh.pos = pos + mesh.u
        fb = linear_interface_force(mesh)

        # lattice visualization
        if interval and t%interval == 0: 
            meshplot(mesh, filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True) 
        
        # next step
        if t >= conv_time:
            if np.sum(np.abs(u_next - u_curr)) <= 1e-8:
                print('Solution converged at time T = %0.4f' % t)
                break
            conv_time += endtime/10
        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        r_curr = ro + u_curr

        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))

    # save data
    mesh.contact.param['fc'] = np.max(np.linalg.norm(interface_force(mesh),2))
    mesh.u = reshape2vector(u_next)
    mesh.pos = pos  # reset to initial configuration
    save_mesh(mesh,'static')
    return mesh    

  
"""Function to implement dynamics of interface"""
def interface_dynamics(mesh, dt, endtime, c = 0, vectorfield = 'off', folder = 'friction', **kwargs):
    # checking the keywords
    if kwargs:
        interval = kwargs['interval']
    else:
        interval = False
    
    if not mesh.contact.param:
        raise Exception("bond formation and breaking parameters are required")

    # update the time step
    mesh.dt = dt

    # saving the mesh object for postprocessing
    if not mesh.folder:
        mesh.folder = fm.create_directory(f"{folder}_{mesh.Ny[0]}X{mesh.Nx[0]}")

    # initial node positions
    pos = copy.deepcopy(mesh.pos)

    # extract displacement boundary conditions
    bcs_comp, bcs_ids, bcs_parser,pin,_,_,_ = mesh.bcs.extract_dbcs()

    # extract load boundary conditions
    lbcs_ids = copy.copy(mesh.lbcs.ids)
    lbcs_fx = copy.copy(mesh.lbcs.fx)
    lbcs_fy = copy.copy(mesh.lbcs.fy)
    lbcs_fun = copy.copy(mesh.lbcs.fun)

    # total number of iterations
    maxsteps = math.ceil(endtime/dt)
    skipsteps = 1
    if maxsteps >= 1e4:
        skipsteps = 10
    mesh.skipsteps = skipsteps  
    save_mesh(mesh) 

    # create A and B matrix without interface bonds
    A = generate_A_matrix(mesh)
    B = generate_B_matrix(mesh)
    # interface force
    mesh.pos = pos + mesh.u
    fb = interface_force(mesh)
    
    # displacement variables for time step 't-1', 't', and 't+1', respectively
    # for verlet algorithm
    u_prev, u_curr, u_next = (flatten(mesh.u) for i in range(3))

    # complementary boundary condition if any
    comp_disp = dispbcs_complementary(u_curr, bcs_comp, bcs_ids)

    # impose boundary condition at t = 0
    u_curr = comp_disp + dispbcs(u_curr,bcs_comp, bcs_ids, bcs_parser, pin, t = 0)
    ro = flatten(pos)
    r_curr = ro +  u_curr
    u_prev = copy.deepcopy(u_curr)

    # writing contact-time data to file
    total_nodes = len(pos)
    file_handler = open(mesh.folder + '/contacts','wb')
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'w')
    dset_u = disp_file.create_dataset(
        name = 'u',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )
    dset_v = disp_file.create_dataset(
        name = 'v',
        shape=(0,total_nodes),
        maxshape = (None, total_nodes),
        dtype='float64',
        compression = 'gzip',
        compression_opts = 9
    )

    # displacement bucket to write to file
    bucket = 0
    fill_steps = 0
    bucket_size = 10000
    bucket_idx = 0
    remain_steps = int(maxsteps/skipsteps)
    if remain_steps < bucket_size:
        bucket_size = remain_steps
    U = np.zeros(shape=(bucket_size, total_nodes))
    V = np.zeros(shape=(bucket_size, total_nodes))
 
    # integration begins
    for step in range(maxsteps):
        # time variable
        t = step*dt

        # velocity of the node
        v = (u_next - u_prev)/(2*dt)

        # impose load boudnary conditions
        load = loadbcs(total_nodes, lbcs_ids, lbcs_fx, lbcs_fy, lbcs_fun, t)
        # time integration
        F = A@r_curr + B - fb - load + c*v
        u_next = 2*u_curr - u_prev - F*dt**2

        # impose boundary conditions
        u_next = comp_disp + dispbcs(u_next, bcs_comp, bcs_ids, bcs_parser, pin, t)

        ###### friction dynamics #######
        mesh.u = reshape2vector(u_next)
        mesh.pos = pos + mesh.u
        

        # macroscopic breaking criterion
        brokenbonds= breaking_condition(mesh)
        # macroscopic new bonds formation condition
        newbonds, new_aph, newlen = forming_condition(mesh,reshape2vector(F),reshape2vector(v))
            
        if brokenbonds or newbonds:
            # add new contact
            contacts = update_contacts(mesh, brokenbonds, newbonds, new_aph, newlen)
            
        # compute interface forces                
        fb = interface_force(mesh)
        # lattice visualization
        if interval and t%interval == 0: 
            meshplot(mesh, filename = f"step_{step}.png", title = f'T = {np.round(t,3)}', vectorfield = vectorfield, save=True) 

        u_prev = copy.deepcopy(u_curr)
        u_curr = copy.deepcopy(u_next)
        r_curr = ro + u_curr

        print('Time step = ',step, 'T = %0.4f' % t, 'Progress = %0.2f' % (100*step/maxsteps))
        if step%skipsteps == 0:
            # saving contact information
            contacts = mesh.contact.contacts
            angles = mesh.contact.angles
            bondlength = mesh.contact.bondlength
            pickle.dump([t, contacts, angles, bondlength], file_handler)
            # saving displacements fields
            u_shape = reshape2vector(u_next)
            U[bucket_idx] = u_shape[:,0]
            V[bucket_idx] = u_shape[:,1] 
            bucket_idx += 1
            if bucket_idx == bucket_size: # if variable is full, empty bucket 
                dset_u.resize(dset_u.shape[0]+bucket_size, axis = 0)
                dset_u[-bucket_size:] = U
                dset_v.resize(dset_v.shape[0]+bucket_size, axis = 0)
                dset_v[-bucket_size:] = V
                bucket += 1
                fill_steps += bucket_size
                remain_steps += -bucket_size 
                bucket_idx = 0
                
                if  remain_steps < bucket_size:
                    bucket_size = remain_steps
                    U = np.zeros(shape = (bucket_size,total_nodes))
                    V = np.zeros(shape = (bucket_size,total_nodes))
                
                
    file_handler.close()
    disp_file.close()

                 