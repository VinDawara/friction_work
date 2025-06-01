import friction.contacts as fc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.collections import LineCollection
import numpy as np
import os
import pickle
import h5py
import copy
import time


"""Field plot in bottom fixed plate"""
def bottom_field(field, extract_data_from, interval = 1, save = False, cbarlim = [], dir = 'fieldplot'):
    # load mesh
    mesh = fc.load_mesh(extract_data_from)

    if field == 'u' or field == 'v':
        # load displacement field
        field_var = np.load(mesh.folder + f'/{field}.npy')

        try:
            max_steps = np.size(field_var,1)
        except:
            max_steps = 1
            field_var = field_var[:,np.newaxis]

        col_index = np.arange(0, max_steps, int(interval/mesh.dt)) 
        tot_col = len(col_index)
         
        # creating grid mesh
        # domain
        xmin, xmax, ymin, ymax = mesh.domain[0]
        # grid points
        Nx = 5*mesh.Nx[0]
        Ny = 5*mesh.Ny[0]
        # mesh grid
        x,y = np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
        field = np.zeros(shape=(tot_col,Ny,Nx)) 
        # total nodes in bottom plate
        bottom_nodes = mesh.Nx[0]*mesh.Ny[0]
        for i in range(tot_col):
            field[i] = griddata(mesh.pos[0:bottom_nodes], field_var[0:bottom_nodes,col_index[i]], (x, y), method='cubic')

        if save:
            path = os.path.join(mesh.folder,f"{dir}/")
            if not os.path.isdir(path):
                os.mkdir(path)
        print(f'Saving plots to {path}')
        for i in range(tot_col):
            t = np.round(col_index[i]*mesh.dt,3)
            fig = plotfield(x,y,field[i], title=f"T = {t}", cbarlim=cbarlim, save=save)
            fig.savefig(path+f"{col_index[i]}", bbox_inches = 'tight', dpi = 300)
        print('Done.')     
            
    return [x,y,field]        
            
"""Function to plot field""" 
def plotfield(x,y,f, title = 'fieldview',xlabel = None, ylabel = None,cbarlim = [], save = False):
    fig, ax = plt.subplots()
    # set axes properties
    if xlabel is not None: 
        ax.set_xlabel(f'{xlabel}')
    if ylabel is not None:    
        ax.set_ylabel(f'{ylabel}')
    ax.set_xlim([np.min(x)-2, np.max(x)+2])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_title(title)
    #ax.set_aspect('equal')
    ax_divider= make_axes_locatable(ax)
    # add an axes to the right of the main axes
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    if not cbarlim:
        ul = np.nanmax(f)
        ll = np.nanmin(f)
        cbarlim = [ll, ul]
        
    surf = ax.pcolormesh(x,y,f,cmap='jet', shading='auto', vmin=cbarlim[0], vmax=cbarlim[1])
    surf.set_clim(cbarlim)
    fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],6),3))
    if save:
        plt.close()
        return fig
    else:
        plt.show()  


"""Function to compute fraction of bonds in contact"""
def intact_bonds(extract_data):
    t = np.array([])
    bond_count = np.array([])
    # load contact information
    f = open(extract_data + f'/contacts', 'rb')
    while True:
        try:
            item = pickle.load(f)
        except:
            print('file reading completed') 
            break  
         
        t =  np.append(t,item[0])
        bond_count = np.append(bond_count, len(item[1]))

    f.close()
    return t, bond_count/bond_count[0]   

"""Function to displacement field along interface"""
def surface_displacments(comp, extract_from, movie = False):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from
    # load displacement data
    try:
        f = np.load(mesh.folder + f'/{comp}.npy')
    except:
        raise KeyError('valid components are \'u\' or \'v\'')

    max_steps = np.size(f,1)
    t = mesh.dt*np.arange(0,max_steps)

    lower_ids = mesh.contact.surfaces[0]
    xdata = mesh.pos[lower_ids,0]
    ydata = np.zeros(shape=(max_steps,len(lower_ids)))

    for i in range(max_steps):
        ydata[i,:] = f[lower_ids,i]

    if movie:
        line_animation(xdata,ydata, t,mesh.folder, f'{comp}')
    return xdata,ydata,t    

"""Function for line animation"""
# This animation is only for line plot where ydata varies with time
def  line_animation(xdata, ydata, time, folder, ylabel = 'y'):
    # round the time float variable
    time = np.round(time,2)

    # create a figure and axes
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # create an empty plot. Note the comma after line variable because it return multiple values
    line, = ax.plot([],[])

    # This function will clear the screen
    def init():
        line.set_data([],[])
        return line,
    
    # initialize the axes
    ax.set_xlabel(r'$x\rightarrow$')
    ax.set_ylabel(r'${%s}\rightarrow$'%ylabel)
    ax.set_xlim([xdata.min(), xdata.max()])
    ax.set_ylim([np.nanmin(ydata), np.nanmax(ydata)])

    # function to update the screen
    def animate(i):
        line.set_data(xdata,ydata[i,:])
        ax.set_title(f'T = {time[i]}')
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(ydata[:,0]), blit = True)

    # saving the animation
    anim.save(folder+f'/{ylabel}.mp4',writer='ffmpeg', fps=1)

"""Function for line animation"""
# This animation is only for line plot where ydata varies with time
def line_movie(xdata, ydata, time, folder, movie_name):
    # round the time float variable
    time = np.round(time,2)

    # create a figure and axes
    fig, ax = plt.subplots()
    # create an empty plot. Note the comma after line variable because it return multiple values
    line, = ax.plot([],[])

    # This function will clear the screen
    def init():
        line.set_data([],[])
        return line,
    
    # initialize the axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([np.min(xdata), np.max(xdata)])
    ax.set_ylim([np.min(ydata), np.max(ydata)])

    # function to update the screen
    def animate(i):
        line.set_data(xdata[i,:],ydata[i,:])
        ax.set_title(f'T = {time[i]}')
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(time), blit = True)

    # saving the animation
    anim.save(folder+f'/{movie_name}.mp4',writer='ffmpeg', fps=500)

"""Function for arrow_vector animations"""
# This animation is only for line and arrow varying
def arrow_vector(xdata, ydata, time, vector, folder, movie_name):
    # normalise the vector
    l = np.sqrt(vector[:,0]**2 + vector[:,1]**2)
    vector[:,0] = np.divide(vector[:,0],l, out = np.zeros_like(vector[:,0]), where = l!=0)
    vector[:,1] = np.divide(vector[:,1],l, out = np.zeros_like(vector[:,1]), where = l!=0)

    # round the time float variable
    time = np.round(time,2)

    # create a figure and axes
    fig, ax = plt.subplots()

    # initialize the axes
    xmin, xmax = np.min(xdata)-1, np.max(xdata)+1
    ymin, ymax = np.min(ydata)-1, np.max(ydata)+1
    ax.set(xlim = (xmin,xmax), ylim = (ymin,ymax))


    # function to update the screen
    def animate(i):
        ax.cla()
        # line.set_data(xdata[i,:],ydata[i,:])
        ax.plot(xdata[i,:],ydata[i,:], color = 'black')
        ax.arrow(xdata[i,1],ydata[i,1], vector[i,0],0, width = 0.01, head_width = 0.1, head_length = 0.2, color = 'blue')  # horizontal vector
        ax.arrow(xdata[i,1],ydata[i,1], 0, vector[i,1], width = 0.01, head_width = 0.1, head_length = 0.2, color = 'green')  # horizontal vector
        ax.set(xlim = (xmin,xmax), ylim = (ymin,ymax))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'T = {time[i]}')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time), blit = False)

    # saving the animation
    anim.save(folder+f'/{movie_name}.mp4',writer='ffmpeg', fps=500)

"""Function for particle animation(Need to test)"""
# This animation is only for particle plot with time
def  particle_motion(x,y, time, folder, movie_name):
    # round the time float variable
    time = np.round(time,2)

    # create a figure and axes
    fig, ax = plt.subplots()
    # create an empty plot. Note the comma after line variable because it return multiple values
    line, = ax.plot([],[], linestyle = 'none', marker = 'o', markersize = 1)
    line2, = ax.plot([],[], linestyle = 'none', marker = 'o', markersize = 2, color = 'red')

    # This function will clear the screen
    def init():
        line.set_data([],[])
        line2.set_data([],[])
        return line, line2,
    
    # initialize the axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    # function to update the screen
    def animate(i):
        line.set_data(x[0:i],y[0:i])
        line2.set_data(x[i],y[i])
        ax.set_title(f'T = {time[i]}')
        return line, line2,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(time), blit = True)

    # saving the animation
    anim.save(folder+f'/{movie_name}.mp4',writer='ffmpeg', fps=500)

"""Function to compute friction force"""
def compute_interface_force(extract_data, comp = 'fx'):
    # load mesh object
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    # load contact information
    ns = mesh.contact.param['kn']
    model = mesh.contact.model 
    fc_dash = mesh.contact.param['fc']

    if model == 'jkr':
        ka = mesh.contact.param['ka'] 
        fa = mesh.contact.param['fa']
        dc = mesh.contact.param['a'] + fa/ka    

    f = open(mesh.folder + '/contacts','rb')

    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 1000
    if maxsteps < bucket_size:
        bucket_size = maxsteps
    
    # output variables    
    t = np.zeros(maxsteps)
    fric = np.zeros(maxsteps)

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]
        contacts = item[1]
        length = item[3]
        if model == 'hertz':
            d0 = np.array(length)
            dc = d0 + (fc_dash/ns)**(2/3)
        elif model == 'double':    
            angles = item[2]
            ts = mesh.contact.param['kt'] 
        elif model == 'double2':
            angles = item[2]
            ts = mesh.contact.param['kt']
            a = mesh.contact.param['a']    
            
        sum = 0
        for i, [id, neigh] in enumerate(contacts):
            id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
            neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
            rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
            lf = np.linalg.norm(rf,2)
            if model == 'linear':
                fb = ns*(lf-length[i])*rf/lf
            elif model == 'hertz':
                fb = (ns*np.abs(lf-dc[i])**(3/2)*np.sign(lf-dc[i]) + fc_dash)*(rf/lf)  
            elif model == 'double':
                ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                dr = np.dot((rf-ri),rf/lf)*rf/lf
                fn = ns*dr
                ft = ts*((rf-ri) - dr)   
                fb = ft + fn
            elif model == 'jkr':
                if lf > length[i]:
                    fs = 0
                else:
                    fs = ns*(lf-length[i])
                pa = ka*(dc - lf)    
                fb = (fs + pa)*rf/lf
            elif model == 'double2':
                ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                fn = ns*(rf[1]-a)
                ft = ts*(rf[0]-ri[0])
                fb = np.array([ft,fn])



            if comp == 'fx':
                fb = fb[0]
            else:
                fb = fb[1]   
            sum += fb
          
        fric[step] = sum  
        bucket_idx += 1
    f.close()    
    disp_file.close()
    end = time.time()
    print('Time elapsed in computing interface forces: ', end-start)
    return t,fric      

"""Function to force on the interface node"""
def interface_bond_force(comp, extract_from, *args):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')

    # load contact information
    ns = mesh.contact.stiff
    model = mesh.contact.model 
    fc_dash = mesh.contact.break_param['fc']

    f = open(mesh.folder + '/contacts','rb')

    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # extracting the node ids of the lower surface corresponding to node number
    upper_ids = mesh.contact.surfaces[1]
    nodes = []
    for arg in args:
        nodes.append(arg+upper_ids[0])

    # output variables    
    t = np.zeros(maxsteps)
    ydata = np.zeros(shape=(len(nodes),maxsteps))

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]
        contacts = item[1]
        length = item[3]
        if model == 'hertz':
            d0 = np.array(length)
            dc = d0 + (fc_dash/ns)**(2/3)

        for j,node in enumerate(nodes):
            for i, [id, neigh] in enumerate(contacts):
                if neigh == node:
                    id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
                    neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                    rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
                    lf = np.linalg.norm(rf,2)
                    if model == 'linear':
                        fb = ns*(lf-length[i])*rf/lf
                    elif model == 'hertz':
                        fb = (ns*np.abs(lf-dc[i])**(3/2)*np.sign(lf-dc[i]) + fc_dash)*(rf/lf)
                    if comp == 'fx':
                        ydata[j,step] = fb[0]
                    elif comp == 'fy':
                        ydata[j,step] = fb[1]   
                    else:
                        ydata[j,step] = np.dot(fb,rf/lf)    
        bucket_idx += 1                     
    f.close()    
    disp_file.close()
    end = time.time()
    print('Time elapsed in computing interface forces: ', end-start)
    return ydata,t    

"""Function to force on the interface node"""
def bond_force_dynamics(node, extract_from):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')

    # load contact information
    ns = mesh.contact.param['kn']
    model = mesh.contact.model 
    fc_dash = mesh.contact.param['fc']

    if model == 'jkr':
        ka = mesh.contact.param['ka'] 
        fa = mesh.contact.param['fa']
        dc = mesh.contact.param['a'] + fa/ka 

    f = open(mesh.folder + '/contacts','rb')

    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # extracting the node ids of the lower surface corresponding to node number
    upper_ids = mesh.contact.surfaces[1]
    node = node + upper_ids[0]

    # output variables    
    t = np.zeros(maxsteps)
    ydata = np.zeros(shape=(3,maxsteps))

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]
        contacts = item[1]
        length = item[3]
        if model == 'hertz':
            d0 = np.array(length)
            dc = d0 + (fc_dash/ns)**(2/3)
        elif model == 'double':    
            angles = item[2]
            ts = mesh.contact.param['kt'] 
        elif model == 'double2':
            angles = item[2]
            ts = mesh.contact.param['kt']
            a = mesh.contact.param['a'] 

        for i, [id, neigh] in enumerate(contacts):
            if neigh == node:
                id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
                neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
                lf = np.linalg.norm(rf,2)
                if model == 'linear':
                    fb = ns*(lf-length[i])*rf/lf
                elif model == 'hertz':
                    fb = (ns*np.abs(lf-dc[i])**(3/2)*np.sign(lf-dc[i]) + fc_dash)*(rf/lf)  
                elif model == 'double':
                    ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                    dr = np.dot((rf-ri),rf/lf)*rf/lf
                    fn = ns*dr
                    ft = ts*((rf-ri) - dr)   
                    fb = np.array([ft,fn])
                elif model == 'jkr':
                    if lf > length[i]:
                        fs = 0
                    else:
                        fs = ns*(lf-length[i])
                    pa = ka*(dc - lf)    
                    fb = (fs + pa)*rf/lf
                elif model == 'double2':
                    ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                    fn = ns*(rf[1]-a)
                    ft = ts*(rf[0]-ri[0])
                    fb = np.array([ft,fn])
                ydata[0,step] = fb[0]
                ydata[1,step] = fb[1]   
                if model == 'double2':
                    ydata[2,step] = np.sqrt(fb[0]**2 + fb[1]**2)
                else:    
                    ydata[2,step] = np.dot(fb,rf/lf)    
        bucket_idx += 1                     
    f.close()    
    disp_file.close()
    end = time.time()
    print('Time elapsed in computing interface forces: ', end-start)
    return ydata,t    

"""Function to force on the interface node"""
def bond_characteristics(node, extract_from):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')

    # load contact information
    ns = mesh.contact.param['kn']
    ts = mesh.contact.param['kt']
    a = mesh.contact.param['a']
    model = mesh.contact.model 

    f = open(mesh.folder + '/contacts','rb')

    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # extracting the node ids of the lower surface corresponding to node number
    upper_ids = mesh.contact.surfaces[1]
    node = node + upper_ids[0]

    # output variables    
    xdata = np.zeros(maxsteps)
    ydata = np.zeros(shape=(3,maxsteps))
    curr_bond = []
    idx = []
    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        contacts = item[1]
        angles = item[2]
        length = item[3]

        for i, [id, neigh] in enumerate(contacts):
            if neigh == node:
                id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
                neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
                lf = np.linalg.norm(rf,2)
                if model == 'linear':
                    fb = ns*(lf-length[i])*rf/lf
                elif model == 'double2':
                    ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                    fn = ns*(rf[1]-a)
                    ft = ts*(rf[0]-ri[0])
                    fb = np.array([ft,fn])

                xdata[step] = lf    
                ydata[0,step] = fb[0]
                ydata[1,step] = fb[1]   
                ydata[2,step] = np.linalg.norm(fb,2)  
                if curr_bond != [id,neigh]:
                    curr_bond = [id,neigh]
                    idx.append(step)  
                    print(curr_bond),print(idx) 
        bucket_idx += 1                     
    f.close()    
    disp_file.close()
    end = time.time()
    print('Time elapsed in computing interface forces: ', end-start)
    # xdata = segment_xaxis(xdata,idx)
    return xdata,ydata  

"""Function to scale the x-axis for cleanliness"""
def segment_xaxis(x,idx,scale = 0.5):
    for i in range(len(idx)-1):
        x[idx[i]:idx[i+1]] += scale*i
    return x    

"""Function to force on the interface node"""
def bond_vector(node, extract_from, movie = True):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')

    # load contact information
    ns = mesh.contact.param['kn']
    model = mesh.contact.model 
    fc_dash = mesh.contact.param['fc']

    if model == 'jkr':
        ka = mesh.contact.param['ka'] 
        fa = mesh.contact.param['fa']
        dc = mesh.contact.param['a'] + fa/ka

    f = open(mesh.folder + '/contacts','rb')

    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # extracting the node ids of the lower surface corresponding to node number
    upper_ids = mesh.contact.surfaces[1]
    node = node + upper_ids[0]

    # output variables    
    t = np.zeros(maxsteps)
    cor = np.zeros(shape=(maxsteps,2))
    force = np.zeros(shape=(maxsteps,2))

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]    
        contacts = item[1]
        length = item[3]
        if model == 'hertz':
            d0 = np.array(length)
            dc = d0 + (fc_dash/ns)**(2/3)
        elif model == 'double':    
            angles = item[2]
            ts = mesh.contact.param['kt'] 
        elif model == 'double2':
            angles = item[2]
            ts = mesh.contact.param['kt']
            a = mesh.contact.param['a']

        for i, [id, neigh] in enumerate(contacts):
            if neigh == node:
                id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
                neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
                cor[step, :] = rf
                lf = np.linalg.norm(rf,2)
                if model == 'linear':
                    force[step,:] = ns*(lf-length[i])*rf/lf
                elif model == 'hertz':
                    force[step,:] = (ns*np.abs(lf-dc[i])**(3/2)*np.sign(lf-dc[i]) + fc_dash)*(rf/lf)
                elif model == 'double':
                    ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                    dr = np.dot((rf-ri),rf/lf)*rf/lf
                    fn = ns*dr
                    ft = ts*((rf-ri) - dr)   
                    force[step,:] = fn+ft
                elif model == 'jkr':
                    if lf > length[i]:
                        fs = 0
                    else:
                        fs = ns*(lf-length[i])
                    pa = ka*(dc - lf)    
                    force[step,:] = (fs + pa)*rf/lf
                elif model == 'double2':
                    ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                    fn = ns*(rf[1]-a)
                    ft = ts*(rf[0]-ri[0])
                    force[step,:] = np.array([ft,fn])    
 
        bucket_idx += 1                     
    f.close()    
    disp_file.close()
    end = time.time()
    print('Time elapsed in computing interface forces: ', end-start)
    if movie == True:
        xdata = np.zeros(shape=(maxsteps,2))
        ydata = np.zeros(shape=(maxsteps,2))
        xdata[:,1] = cor[:,0]
        ydata[:,1] = cor[:,1]
        arrow_vector(xdata, ydata, t, force, mesh.folder, 'vector')
        print(f'movie vector.mp4 saved to {mesh.folder}')
  


"""Function to plot the trajectory of multiple point"""
def trajectory(x,t,d, scale = 100):
    fig, ax = plt.subplots()
    for i, xo in enumerate(x):
        ax.plot(t,xo+scale*d[:,i], linestyle = 'solid', color = 'black')
    ax.set_xlabel('time')
    ax.set_ylabel('x')  
    plt.show()   

    

"""Function to compute the velocity field on the top plate"""
def compute_velocity_top(field, extract_data_from, interval = 1, save = False, cbarlim = [], dir = 'fieldplot'):
    # load  mesh object
    mesh = fc.load_mesh(extract_data_from) 
    dt = mesh.dt
    # load field variable
    try:
        f = np.load(mesh.folder + f'/{field}.npy')
    except:
        raise SyntaxError('can take \'u\' or \'v\' as an argument')

    # max number of time steps
    maxsteps = np.size(f,1)  
    # interested time index  
    time_idx = np.arange(0, maxsteps, int(interval/dt)) 
    tot_plots = len(time_idx)
    # compute velocity
    v = np.zeros(np.shape(f))
    for i in range(maxsteps):
        if i == 0:
            v[:,i] = (f[:,i+1] - f[:,i])/dt
        elif i == maxsteps-1:
            v[:,i] = (f[:,i] - f[:,i-1])/dt    
        else:
            v[:,i] = (f[:,i+1] - f[:,i-1])/(2*dt)
    # domain
    xmin, xmax, ymin, ymax = mesh.domain[1]   
    # grid points
    Nx = 4*mesh.Nx[1]
    Ny = 4*mesh.Ny[1]
    x,y  = np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))
    field = np.zeros(shape=(tot_plots,Ny,Nx))
    # number of nodes on top plate
    top_nodes = mesh.Nx[1]*mesh.Ny[1]
    for i in range(tot_plots):
        field[i] = griddata(mesh.pos[-top_nodes:-1], v[-top_nodes:-1,time_idx[i]], (x, y), method='cubic')
        
    if save:
        path = os.path.join(mesh.folder,f"{dir}/")
        if not os.path.isdir(path):
             os.mkdir(path)
        print(f'Saving plots to {path}')
        for i in range(tot_plots):
            t = np.round(time_idx[i]*mesh.dt,3)
            fig = plotfield(x,y,field[i], title=f"T = {t}", cbarlim=cbarlim, save=save)
            fig.savefig(path+f"{time_idx[i]}", bbox_inches = 'tight', dpi = 300)
        print('Done.')     
            
    return [x,y,field]   



# compute normal force and coefficient of friction

"""Function to extract spatial position of the intact bonds"""
def bond_spatial_position(extract_from, end = np.inf):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from
    top_ids = mesh.contact.surfaces[1]
    state = np.empty(shape = (1,len(top_ids)), dtype=int)
    t = np.array([])
    x = mesh.pos[top_ids,0]
    # load contact information
    f = open(mesh.folder + f'/contacts', 'rb')
    indicator = 0
    while True:
        try:
            item = pickle.load(f)
            curr_state = np.zeros(shape = (1,len(top_ids)),dtype=int)
        except:
            print('file reading completed') 
            break  
        
        if item[0] >= end:
            break 
        t =  np.append(t,item[0])
        interface_bonds = item[1]
        # ids = [id-top_ids[0] for _,id in interface_bonds]
        # curr_state[0,ids] = 1

        for _,id in interface_bonds:
            curr_state[0,id-top_ids[0]] += 1

        if indicator:
            state = np.append(state,curr_state, axis=0)
        else:
            state = curr_state    
            indicator += 1
  
            
    f.close()
    return t,x,state

"""Spartial force-time graph"""
def spatial_force_time(extract_data):
    # load mesh object
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps
    
    # load contact information
    ns = mesh.contact.param['kn']
    model = mesh.contact.model 
    fc_dash = mesh.contact.param['fc']
    top_nodes_ids = mesh.contact.surfaces[1]
    if model == 'jkr':
        ka = mesh.contact.param['ka'] 
        fa = mesh.contact.param['fa']
        dc = mesh.contact.param['a'] + fa/ka 
    # get x position of top surface
    x = mesh.pos[top_nodes_ids,0]
    
    f = open(mesh.folder + '/contacts','rb')

    t = np.zeros(maxsteps)
    fric = np.zeros(shape = (maxsteps,len(top_nodes_ids)))

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]
        contacts = item[1]
        length = item[3]

        if model == 'hertz':
            d0 = np.array(length)
            dc = d0 + (fc_dash/ns)**(2/3)
        elif model == 'double':    
            angles = item[2]
            ts = mesh.contact.param['kt'] 
        elif model == 'double2':
            angles = item[2]
            ts = mesh.contact.param['kt']
            a = mesh.contact.param['a']

        for i, [id, neigh] in enumerate(contacts):
            id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
            neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
            rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
            lf = np.linalg.norm(rf,2)
            if model == 'linear':
                fb = ns*(lf-length[i])*rf[0]/lf
            elif model == 'hertz':
                fb = (ns*np.abs(lf-dc[i])**(3/2)*np.sign(lf-dc[i]) + fc_dash)*(rf[0]/lf)  
            elif model == 'double':
                ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                dr = np.dot((rf-ri),rf/lf)*rf/lf
                fn = ns*dr
                ft = ts*((rf-ri) - dr)   
                fb = ft
            elif model == 'jkr':
                if lf > length[i]:
                    fs = 0
                else:
                    fs = ns*(lf-length[i])
                pa = ka*(dc - lf)    
                fb = (fs + pa)*rf[0]/lf
            elif model == 'double2':
                ri = length[i]*np.array([np.cos(np.pi*angles[i]/180),np.sin(np.pi*angles[i]/180)])
                fn = ns*(rf[1]-a)
                ft = ts*(rf[0]-ri[0])
                fb = ft

            fric[step,neigh-top_nodes_ids[0]] += fb
        bucket_idx += 1    
                  
    f.close()   
    disp_file.close()
    end = time.time()
    print('Time elpased: ', end-start)
    return t,x,fric 


"""Function to generate the list of broken and new bonds dynamics"""
def generate_bond_dynamics(folder):
    
    f = open(folder + '/contacts','rb')
    b = open(folder + '/brokenbonds','wb')
    n = open(folder + '/newbonds','wb')
    # reading initial contacts
    item = pickle.load(f)
    curr_contacts = item[1]
    while True:
        brokenbonds = []
        newbonds = []
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')
            break

        t = item[0]
        new_contacts = item[1]
        if curr_contacts != new_contacts:
            for id,neigh in curr_contacts:
                if [id,neigh] not in new_contacts:
                    brokenbonds.append([id,neigh])
            for id,neigh in new_contacts:
                if [id,neigh] not in curr_contacts:
                    newbonds.append([id,neigh])    
        curr_contacts = copy.deepcopy(new_contacts)
        if brokenbonds:
            pickle.dump([t,brokenbonds],b)  
        if newbonds:    
            pickle.dump([t,newbonds], n)              

    f.close()   
    b.close()
    n.close()

                
"""Function to extract spatial position of the intact bonds"""
def brokenbonds_spatial_position(extract_from):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from
    top_ids = mesh.contact.surfaces[1]
    state = np.empty(shape = (1,len(top_ids)), dtype=int)
    t = np.array([])
    x = mesh.pos[top_ids,0]
    # load brokenbonds information
    f = open(mesh.folder + f'/contacts', 'rb')
    b = open(mesh.folder + f'/brokenbonds','rb')
    indicator = 0
    item2 = pickle.load(b)
    time = item2[0]
    interface_bonds = item2[1]
    while True:
        try:
            item = pickle.load(f)
            curr_state = np.zeros(shape = (1,len(top_ids)),dtype=int)
        except:
            print('file reading completed') 
            break  
         
        t =  np.append(t,item[0])
        if time == item[0]:
            for _,id in interface_bonds:
                curr_state[0,id-top_ids[0]] += 1

            try:
                item2 = pickle.load(b)
                time = item2[0]
                interface_bonds = item2[1]
            except:    
                print('broken bonds file reading completed') 
                

        if indicator:
            state = np.append(state,curr_state, axis=0)
        else:
            state = curr_state    
            indicator += 1
                      
            
    f.close()
    b.close()
    return t,x,state


"""Function to read the data"""
def read_data(extract_from):
  
    # load brokenbonds information
    f = open(extract_from + f'/brokenbonds','rb')

    # empty list
    data = []

    while True:
        try:
            item = pickle.load(f)
            data.append(item)
        except:
            print('file reading completed') 
            break  
                          
    f.close()
    return data

"""Function to extract trajectory of the crack"""
def generate_crack_trajectory(extract_data, crack = 'broken'):
    # load mesh object
    mesh = fc.load_mesh(extract_data)
    # update directory
    mesh.folder = extract_data

    # extract x position of the interface nodes of the top plate
    top_nodes_ids = mesh.contact.surfaces[1]
    x = mesh.pos[top_nodes_ids,0]
    xmin = np.min(x)
    xmax = np.max(x)

    # open the file  name 'brokenbonds' for reading to extract interface broken bond information
    if crack == 'new':
        file = 'newbonds'
    else:
        file = 'brokenbonds'

    f = open(mesh.folder + f'/{file}','rb')
    time = []
    xpos = []
    direction = []
    # read the data one-by-one
    while True:
        try:
            data = pickle.load(f)
        except:
            print('File reading completed')
            break

        t = data[0]
        bond_ids = data[1]
        pos = np.zeros(len(bond_ids))
        for i,[_,id] in enumerate(bond_ids):
            pos[i] = x[id-top_nodes_ids[0]]

        # scan this position bucket one by one (algorithm)
        # 1. Pre-scanning stage: Take a crack event from the bucket list, and find to how many curves it is closest
        # based on lattice distance, specifically, check the distance between a new crack event with existing crack 
        # tip location of the curves. Filter how many are closest based on unit time spacing. 
        # If none, increase the time spacing until found one. Make a checklist of the curve index.
        # (a crack event may be part of one of these curves)

        # 2. If checkist contains more than one curve, then find the one curve having highest time.
        # Check the direction of the new crack event with this potential curves. If direction is zero,
        # then add the event to this curve. Else, if direction is nonzero then add it may add depending on
        # the satisfication of direction condition of the potential curve or existing
        # direction is zero.

        # 3. If checklist contains none, then create new curve.
        # 4. Scan the crack tip of curves, if direction  is -1 and tip is at left end. It stops, or vice-versa

        
        for i in pos:
            new_curve = False
            proximity_idx = []  # index of the curves satifying proximity condition

            # pre-scanning stage 1: proximity check
            for k in range(len(xpos)):
                if xpos[k][-1] == xmin and direction[k] == -1:
                    pass
                elif xpos[k][-1] == xmax and direction[k] == 1:
                    pass
                else:
                    direction[k] = path_direction(xpos[k])
                    if abs(xpos[k][-1] -i) <= 1:
                        proximity_idx.append(k)

            # pre-scanning stage 2: time check        
            potential_curve_idx = []
            dt = 1
            while True: 
                for k in proximity_idx:
                    if t - time[k][-1] <= dt:
                        potential_curve_idx.append(k)
                if potential_curve_idx:
                    break        
                else:
                    if dt<=5000:    #300#5000
                        dt += 1
                    else:
                        break            

            # scanning stage 1: 
            if potential_curve_idx:
                # Find curve having maximum time
                max_time = -1
                for k in potential_curve_idx:
                    if time[k][-1] >= max_time:
                        idx_max = k
                        max_time = time[k][-1]
                # check direction condition        
                dir = i-xpos[idx_max][-1]
                if dir:
                    if direction[idx_max]:
                        if np.sign(dir) == direction[idx_max]:
                            xpos[idx_max] += [i]
                            time[idx_max] += [t]   
                    else:
                        xpos[idx_max] += [i]
                        time[idx_max] += [t] 

                else:
                    xpos[idx_max] += [i]
                    time[idx_max] += [t]   
                    # new_curve = True 

            else:
                new_curve = True            

            # for k in range(len(xpos)):
            #     if abs(xpos[k][-1]-i) <= 1.01 and t-time[k][-1]<=6:
            #         direction[k] = path_direction(xpos[k]) 
            #         if direction[k]:
            #             if np.sign(i - xpos[k][-1]) == direction[k]:
            #                 xpos[k] += [i]
            #                 time[k] += [t]
            #                 new_curve = False 
            #         else:
            #             xpos[k] += [i]
            #             time[k] += [t]
            #             new_curve = False  
                        
            if new_curve:
                xpos.append([i])
                time.append([t]) 
                direction.append([0])        

  
    f.close()
    return time, xpos                             

"""Function to obtain path direction"""
def path_direction(x):
    xo = x[0]
    for i in x:
        if np.sign(i-xo)>0:
            return 1
        elif np.sign(i-xo)<0:
            return -1
    return 0    
  
"""Function to extract scatter plot for broken bonds"""
def get_event_spiotemp_info(extract_from, crack = 'all'):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from
    top_ids = mesh.contact.surfaces[1]
    x = mesh.pos[top_ids,0]

    t = np.array([])
    pos = np.array([])
    # load brokenbonds information
    if crack == 'new':
        file = 'newbonds'
    elif crack == 'broken':
        file = 'brokenbonds'
    else:
        file = 'contacts'    

    f = open(mesh.folder + f'/{file}','rb')

    while True:
        try:
            item = pickle.load(f)
        except:
            print('file reading completed') 
            break  
          
        bond_ids = item[1]
        for _,id in bond_ids:
            t =  np.append(t,item[0])
            pos = np.append(pos,x[id-top_ids[0]])
           
    f.close()

    return t,pos

"""Function to compute velocity from the trajectory"""
def crack_velocity(t,x):
    f = interp1d(t,x)
    N = 100*len(t)
    tn = np.linspace(t.min(), t.max(),N)
    dt = tn[1]-tn[0]
    xn = f(tn)
    v = np.zeros(N)
    for i in range(N):
        if i==N-1:
            v[i] = (xn[i-2]-4*xn[i-1] + 3*xn[i])/(2*dt)

        elif i==0:
            v[i] = (4*xn[i+1]-xn[i+2] - 3*xn[i])/(2*dt)

        else:
            v[i] = (xn[i+1]-xn[i-1])/(2*dt)   
  
    return tn,xn,v           

"""Function to view the mesh during post-processing"""
def meshplot(extract_data, t, save = False):
    # load mesh
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps
    pos = copy.deepcopy(mesh.pos)

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    # handler for interface bonds data file
    inter_file = open(mesh.folder + '/contacts', 'rb')
    
    # time steps corresponding to the time intervals
    time_steps = np.rint(t/(dt*skipsteps))    
    t = time_steps*skipsteps*dt

    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0    
        try:
            item = pickle.load(inter_file)
        except:
            print('file reading completed')   
            break
        if np.any(t == item[0]):
            mesh.pos[:,0] = pos[:,0] + u[bucket_idx,:]
            mesh.pos[:,1] = pos[:,1] + v[bucket_idx,:]
            mesh.contact.contacts = item[1]
            fc.meshplot(mesh, save=save, filename=f'{step}', title=f't={item[0]}', dir = 'mesh') 
        elif item[0] > np.max(t):
            print('Data reading completed')
            break       
        bucket_idx += 1

"""Function to displacement field along interface"""
def node_life(comp, extract_from, *args):
    # load mesh object
    mesh = fc.load_mesh(extract_from)
    mesh.folder = extract_from
    dt = mesh.dt
    # load displacement data
    try:
        f = np.load(mesh.folder + f'/{comp}.npy')
    except:
        raise KeyError('valid components are \'u\' or \'v\'')

    maxsteps = np.size(f,1)
    t = mesh.dt*np.arange(0,maxsteps)
    # compute velocity
    v = np.zeros(np.shape(f))
    for i in range(maxsteps):
        if i == 0:
            v[:,i] = (f[:,i+1] - f[:,i])/dt
        elif i == maxsteps-1:
            v[:,i] = (f[:,i] - f[:,i-1])/dt    
        else:
            v[:,i] = (f[:,i+1] - f[:,i-1])/(2*dt)

    upper_ids = mesh.contact.surfaces[1]
    # reading the node number of the upper surface start from zero
    nodes = []
    for arg in args:
        nodes.append(arg+upper_ids[0])

    ydata = np.zeros(shape=(len(nodes),maxsteps))

    for i in range(maxsteps):
        ydata[:,i] = v[nodes,i]

    return ydata,t    

"""Function to compute friction force"""
def compute_boundary_force(extract_data, comp, boundary):
    # load mesh object
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps
    neighbors = mesh.neighbors
    norm_stiff = mesh.normal_stiffness
    tang_stiff = mesh.tangential_stiffness
    angles = mesh.angles

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # output variables    
    t = np.zeros(maxsteps)
    f = np.zeros(maxsteps)
    # set boundary
    if boundary == 'top':
        ids = mesh.top[1]
    elif boundary == 'right':
        ids = mesh.right[1]
    elif boundary == 'bottom':    
        ids = mesh.bottom[1]  
    elif boundary == 'left':
        ids = mesh.left[1]      

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0

        t[step] = skipsteps*step*dt
        sum = 0
        for id in ids:
            id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
            for i,neigh in enumerate(neighbors[id]):
                ns = norm_stiff[id][i]
                ts = tang_stiff[id][i]
                aph = angles[id][i]               
                neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
                ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
                rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
                fb = (ns-ts)*np.dot(rf-ro,ro)*ro + ts*(rf-ro)
                if comp == 'fx':
                    sum += fb[0]
                else:
                    sum += fb[1]    
          
        f[step] = -sum  
        bucket_idx += 1
    end = time.time()
    print('Time elaspsed in computing boundary forces: ', end-start)      
    return t,f      

"""Function to compute the strain fields"""
def strainfield(comp, extract_data, start = 0,end = None,step = 1, range = None, cbar = None):
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps       # for current data, remove factor 10 in future
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    nlow_nodes = mesh.Nx[0]*mesh.Ny[0]

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    # handler for interface bonds data file
    inter_file = open(mesh.folder + '/contacts', 'rb')
    
    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((range/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # create strain directory insider folder
    sub_dir = os.path.join(mesh.folder,f'strain{comp}/')
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break   
        nx = 2*mesh.Nx[0]
        ny = 2*mesh.Ny[0]
        xl,yl,low_strain = straintensor(pos[0:nlow_nodes,:],u[bucket_idx,0:nlow_nodes],
                                     v[bucket_idx,0:nlow_nodes], nx, ny, comp)
        nx = 2*mesh.Nx[1]
        ny = 2*mesh.Ny[1]
        xu,yu,up_strain = straintensor(pos[nlow_nodes:-1,:],u[bucket_idx,nlow_nodes:-1],
                                     v[bucket_idx,nlow_nodes:-1], nx, ny, comp)
        # extracting interface bonds corresponding to the time step
        while True:
            try:
                inter_data = pickle.load(inter_file)
            except:
                print('Interface file reading completed')       
            if inter_data[0]==t[i]:
                break
          
        inter_bonds = inter_data[1]
        segs = np.zeros(shape = (len(inter_bonds), 2,2))
        new_pos = pos + np.column_stack((u[bucket_idx,:],v[bucket_idx,:]))
        # create a line segments
        for j, [id1, id2] in enumerate(inter_bonds):
            x = (new_pos[id1,0], new_pos[id2,0])
            y = ([new_pos[id1,1], new_pos[id2,1]])
            segs[j,:,0] = x
            segs[j,:,1] = y

        line_collection = LineCollection(segs, linewidths = 0.5  , color = 'black')

        # visualization
        fig, ax = plt.subplots()
        ax_divider= make_axes_locatable(ax)
        # add an axes to the right of the main axes
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        if cbar is None:
            ul = max(np.nanmax(low_strain),np.nanmax(up_strain))
            ll = min(np.nanmin(low_strain),np.nanmin(up_strain))
            cbarlim = [ll, ul] 
        else:
            cbarlim = cbar    
        ax.set_title(f't = {t[i]}')
        ax.set_xlim([mesh.domain[0,0]-2, mesh.domain[0,1]+2])
        ax.set_ylim([mesh.domain[0,2], mesh.domain[1,3]+2])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')

        surf = ax.pcolormesh(xl,yl, low_strain, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1] )
        surf = ax.pcolormesh(xu, yu, up_strain, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1])
        ax.add_collection(line_collection)
        fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],4),3))
        fig.savefig(sub_dir+f"{int(step)}.png", bbox_inches = 'tight', dpi = 300)
        plt.close()               
   
    disp_file.close()
    inter_file.close()

    

"""compute nodal strain tensor"""
def computestraintensor(mesh):
    neighbors = mesh.neighbors
    angles = mesh.angles
    u = mesh.u
    total_nodes = len(mesh.pos)
    strain = np.zeros(shape=(total_nodes,4))
    for id in range(total_nodes):
        s = 0
        for i,neigh in enumerate(neighbors[id]):
            aph = angles[i]
            ro = np.array([np.cos(np.pi*aph/180), np.sin(np.pi*aph/180)])
            uij = mesh.u[neigh] - mesh.u[id]
            s += (1/6)*(uij[:,np.newaxis]@ro[np.newaxis,:] + \
                    ro[:,np.newaxis]@uij[np.newaxis,:])
            strain[id] = s.reshape(1,4) 
    return strain        


"""first derivative operator"""
def d_opr(l):
    # derivative operator
    dir_opr = np.zeros(shape=(l,l))
    for i in range(l):
        if i==0:
            dir_opr[i,i+1] = 1
            dir_opr[i,i] = -1
        elif i==l-1:
            dir_opr[i,i] = 1
            dir_opr[i,i-1] = -1
        else:
            dir_opr[i,i+1] = 0.5
            dir_opr[i,i-1] = -0.5
    return dir_opr        


"""compute first derivative in a regular grid"""
def computefirstderivative(v,dx,dy):
    r,c = v.shape

    # x derivative 
    dir_opr = d_opr(c)
    vx = np.zeros(v.shape)
    for i in range(r):
        vx[i,:] = (1/dx)*dir_opr@v[i,:]

    # y derivative
    dir_opr = d_opr(r)
    vy = np.zeros(v.shape)
    for i in range(c):
        vy[:,i] = (1/dy)*dir_opr@v[:,i]

    return vx,vy   

"""compute strain tensor from the displacement fields""" 
def straintensor(pos,u,v,nx,ny, comp, coord = 'deformed'):
    # interpolate along regular grid
    # create mesh grid
    xmin, xmax = np.min(pos[:,0])+1, np.max(pos[:,0])-1
    ymin, ymax = np.min(pos[:,1])+0.01, np.max(pos[:,1])-0.3
    x,y = np.meshgrid(np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax, ny))
    up = griddata(pos, u, (x,y), method='cubic')
    vp = griddata(pos, v, (x,y), method='cubic')
    
    # compute first derivative
    # grid spacing
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]

    if coord == 'deformed':
        x += up
        y += vp

    ux,uy = computefirstderivative(up,dx,dy)
    if comp == 'xx':
        return x,y,ux
    vx,vy = computefirstderivative(vp,dx,dy)
    if comp == 'yy':
        return x,y,vy
    else:
        return x,y,0.5*(uy+vx)

"""Function to compute the stress fields"""
def stressfield(comp, extract_data,poisson = 0.2, start = 0,end = None,step = 1, range = None, cbar = None):
    # path = r'c:\\Users\Admin\Documents\\VScode\\scripts\\friction'
    # plt.style.use(path + './stress.mplstyle')
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps       # for current data, remove factor 10 in future
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    nlow_nodes = mesh.Nx[0]*mesh.Ny[0]

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    # handler for interface bonds data file
    inter_file = open(mesh.folder + '/contacts', 'rb')
    
    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((range/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # create strain directory insider folder
    sub_dir = os.path.join(mesh.folder,f'stress_cont{comp}/')
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break   
        nx = 2*mesh.Nx[0]
        ny = 2*mesh.Ny[0]
        xl,yl,low_stress = stresstensor(poisson, pos[0:nlow_nodes,:],u[bucket_idx,0:nlow_nodes],
                                     v[bucket_idx,0:nlow_nodes], nx, ny, comp, coord='deformed')
        nx = 2*mesh.Nx[1]
        ny = 2*mesh.Ny[1]
        xu,yu,up_stress = stresstensor(poisson,pos[nlow_nodes:-1,:],u[bucket_idx,nlow_nodes:-1],
                                     v[bucket_idx,nlow_nodes:-1], nx, ny, comp,coord='deformed')
        # extracting interface bonds corresponding to the time step
        while True:
            try:
                inter_data = pickle.load(inter_file)
            except:
                print('Interface file reading completed')       
            if inter_data[0]==t[i]:
                break
          
        inter_bonds = inter_data[1]
        segs = np.zeros(shape = (len(inter_bonds), 2,2))
        new_pos = pos + np.column_stack((u[bucket_idx,:],v[bucket_idx,:]))
        # create a line segments
        for j, [id1, id2] in enumerate(inter_bonds):
            x = (new_pos[id1,0], new_pos[id2,0])
            y = ([new_pos[id1,1], new_pos[id2,1]])
            segs[j,:,0] = x
            segs[j,:,1] = y

        line_collection = LineCollection(segs, linewidths = 1  , color = 'tab:red')

        # visualization
        fig, ax = plt.subplots(figsize = (10,10))
        ax_divider= make_axes_locatable(ax)
        # add an axes to the right of the main axes
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        if cbar is None:
            ul = max(np.nanmax(low_stress),np.nanmax(up_stress))
            ll = min(np.nanmin(low_stress),np.nanmin(up_stress))
            cbarlim = [ll, ul] 
        else:
            cbarlim = cbar    
        ax.set_title(r'$t = {%s}$'%t[i], fontsize = 20)
        ax.set_xlim([mesh.domain[0,0]-2, mesh.domain[0,1]+2])
        ax.set_ylim([mesh.domain[0,2], mesh.domain[1,3]+2])
        ax.set_xlabel(r'$x/a\rightarrow$')
        ax.set_ylabel(r'$y/a\rightarrow$')
        ax.set_aspect('equal')
        # ax.yaxis.set_tick_params(labelbottom=False)
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.set_xticks([])
        # ax.set_yticks([])

        surf = ax.pcolormesh(xl,yl, low_stress, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1] )
        surf = ax.pcolormesh(xu, yu, up_stress, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1])
        # cont = ax.contourf(xl,yl, low_stress, levels = 20,cmap = 'jet',vmin=cbarlim[0], vmax=cbarlim[1])
        # cont = ax.contourf(xu, yu, up_stress, levels = 20,cmap = 'jet',vmin=cbarlim[0], vmax=cbarlim[1])
        ax.add_collection(line_collection)
        fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],4),3))
        fig.savefig(sub_dir+f"{int(step)}", bbox_inches = 'tight', dpi = 300)
        plt.close()               
   
    disp_file.close()
    inter_file.close()

"""compute stress tensor from the displacement fields""" 
def stresstensor(nu, pos,u,v,nx,ny, comp, coord = 'deformed'):
    # interpolate along regular grid
    # create mesh grid
    # earlier +0.5,-0.5 for x, and 0.01,-0.3 for y
    xmin, xmax = np.min(pos[:,0])+1, np.max(pos[:,0])-1
    ymin, ymax = np.min(pos[:,1])+0.01, np.max(pos[:,1])-0.3
    x,y = np.meshgrid(np.linspace(xmin,xmax,nx), np.linspace(ymin,ymax, ny))
    up = griddata(pos, u, (x,y), method='cubic')
    vp = griddata(pos, v, (x,y), method='cubic')
    
    # compute first derivative
    # grid spacing
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]

    ux,uy = computefirstderivative(up,dx,dy)
    vx,vy = computefirstderivative(vp,dx,dy)

    if coord == 'deformed':
        x += up
        y += vp

    if comp == 'xx':
        sx = (1/(1+nu))*(ux + nu/(1-2*nu)*(ux+vy))
        return x,y,sx
    
    if comp == 'yy':
        sy = (1/(1+nu))*(vy + nu/(1-2*nu)*(ux+vy))
        return x,y,sy
    else:
        sxy = (1/(1+nu))*0.5*(uy+vx)
        return x,y,sxy
    
"""Extract stress field along the interface (we don't need to interpolate everywhere)"""    
def interface_stress(comp, extract_data, plate,poisson = 0.2, start = 0,end = None,step = 1, range = None):
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps      
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    nlow_nodes = mesh.Nx[0]*mesh.Ny[0]
    if plate == 'lower':
        nx = 5*mesh.Nx[0]
        ny = 5
        source = nlow_nodes-3*mesh.Nx[0]
        target = nlow_nodes  # this point will be excluded
    elif plate == 'upper':
        nx = 5*mesh.Nx[1]
        ny = 5
        source = nlow_nodes
        target = nlow_nodes + 3*mesh.Nx[1]   # last point will be excluded
    else:
        raise Exception(f'Invalid argument: \'{plate}\' does not recognized')
    
    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np. rint((range/(dt*skipsteps)))   

    t = time_steps*skipsteps*dt
    field = np.zeros((len(t),nx))

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break    
        x,_,stress = stresstensor(poisson, pos[source:target,:],u[bucket_idx,source:target],
                                     v[bucket_idx,source:target], nx, ny, comp, coord='undeformed')
        
        if plate == 'lower':
            field[i,:] = stress[-1,:]
        else:
            field[i,:] = stress[0,:]    
        
    disp_file.close()
    x = x[0,:]
    return t,x,field

"""Extract stress field along the interface (we don't need to interpolate everywhere)"""    
def interface_strain(comp, extract_data, plate, start = 0,end = None,step = 1, range = None):
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps      
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    nlow_nodes = mesh.Nx[0]*mesh.Ny[0]
    if plate == 'lower':
        nx = 5*mesh.Nx[0]
        ny = 5
        source = nlow_nodes-3*mesh.Nx[0]
        target = nlow_nodes  # this point will be excluded
    elif plate == 'upper':
        nx = 5*mesh.Nx[1]
        ny = 5
        source = nlow_nodes
        target = nlow_nodes + 3*mesh.Nx[1]   # last point will be excluded
    else:
        raise Exception(f'Invalid argument: \'{plate}\' does not recognized')
    
    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np. rint((range/(dt*skipsteps)))   

    t = time_steps*skipsteps*dt
    field = np.zeros((len(t),nx))

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break    
        x,_,stress = straintensor(pos[source:target,:],u[bucket_idx,source:target],
                                     v[bucket_idx,source:target], nx, ny, comp, coord='undeformed')
        
        if plate == 'lower':
            field[i,:] = stress[-1,:]
        else:
            field[i,:] = stress[0,:]    
        
    disp_file.close()
    x = x[0,:]
    return t,x,field

"""Displacement of the nodes of the top row"""
def interface_nodal_disp(comp, extract_data, plate, start = 0,end = None,step = 1, range = None, va=0):
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    skipsteps = mesh.skipsteps 
    dt = mesh.dt    
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    if plate == 'lower':
        nx = 5*mesh.Nx[0]
        ids = mesh.contact.surfaces[0]
    elif plate == 'upper':
        nx = 5*mesh.Nx[1]
        ids = mesh.contact.surfaces[1]
    else:
        raise Exception(f'Invalid argument: \'{plate}\' does not recognized')
    
    # create query points
    x = pos[ids,0]
    xmin, xmax = x.min(), x.max()
    xq = np.linspace(xmin,xmax,nx)

    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np. rint((range/(dt*skipsteps)))   

    t = time_steps*skipsteps*dt
    disp = np.zeros((len(t),nx))

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file[f'{comp}'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step
                if remain_step < bucket_size:
                    bucket_size = remain_step
            
            bucket_idx = int(step - idx)   
            if bucket_idx < fill_step-idx:
                break 

        u_curr = u[bucket_idx,ids]
        f = interp1d(x,u_curr, fill_value='extrapolate')
        disp[i,:] = f(xq) - va*t[i]
                          
    disp_file.close()
    return t,xq,disp

"""Sliding velocity of the nodes of the top row"""
def interface_nodal_velocity(comp, extract_data, plate, start = 0,end = None,step = 1, range = None):
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    skipsteps = mesh.skipsteps 
    dt = mesh.dt    
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    if plate == 'lower':
        nx = 5*mesh.Nx[0]
        ids = mesh.contact.surfaces[0]
    elif plate == 'upper':
        nx = 5*mesh.Nx[1]
        ids = mesh.contact.surfaces[1]
    else:
        raise Exception(f'Invalid argument: \'{plate}\' does not recognized')
    
    # create query points
    x = pos[ids,0]
    xmin, xmax = x.min()+1, x.max()-1
    xq = np.linspace(xmin,xmax,nx)

    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np. rint((range/(dt*skipsteps)))   

    t = time_steps*skipsteps*dt
    velocity = np.zeros((len(t),nx))

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    dt = dt*skipsteps
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file[f'{comp}'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
            
            bucket_idx = int(step - idx)  
            if bucket_idx < fill_step-idx:
                break 

        u_curr = u[bucket_idx,ids]
        if bucket_idx == bucket_size-1:
            if step != maxsteps:
                u_next = disp_file[f'{comp}'][fill_step+bucket_size] 
                u_next = u_next[ids] 
            else:
                u_prev = u[bucket_idx-1,ids]   
        else:
            u_next = u[bucket_idx+1,ids]

        if step == maxsteps:
            f = interp1d(x,u_curr)
            u_curr = f(xq)
            f = interp1d(x,u_prev)
            u_prev = f(xq)
            velocity[i,:] = (u_curr-u_prev)/dt     
        else:
            f = interp1d(x,u_next)
            u_next = f(xq)
            f = interp1d(x,u_curr)
            u_curr = f(xq)
            velocity[i,:] = (u_next-u_curr)/dt
                           
    disp_file.close()
    return t,xq,velocity



"""Function to compute the stress fields"""
def stressfield_paper(comp, extract_data,poisson = 0.2, start = 0,end = None,step = 1, range = None, cbar = None):
    path = r'c:\\Users\Admin\Documents\\VScode\\scripts\\friction'
    plt.style.use(path + './stress.mplstyle')
    # load mesh data
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data
    dt = mesh.dt
    skipsteps = mesh.skipsteps      
    pos = copy.deepcopy(mesh.pos)

    # number of nodes on lower lattice
    nlow_nodes = mesh.Nx[0]*mesh.Ny[0]

    # handler for displacement data file
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    # handler for interface bonds data file
    inter_file = open(mesh.folder + '/contacts', 'rb')
    
    # time steps corresponding to the time intervals
    if range is None:
        time_steps = np.rint((np.arange(start,end,step)/(dt*skipsteps)))
    else:
        time_steps = np.rint((range/(dt*skipsteps)))    
    t = time_steps*skipsteps*dt

    # preparing the batch reading for displacement data
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = int(time_steps[0])
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps

    # create strain directory insider folder
    sub_dir = os.path.join(mesh.folder,f'stress{comp}/')
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
   
    
    for i,step in enumerate(time_steps):
        # read displacement data in buckets
        while True:
            if step >= fill_step:
                u = disp_file['u'][fill_step:fill_step+bucket_size]
                v = disp_file['v'][fill_step:fill_step+bucket_size]
                bucket += 1
                idx = fill_step
                fill_step += bucket_size
                remain_step = maxsteps - fill_step 
                if remain_step < bucket_size:
                    bucket_size = remain_step
                print('Data extracted:%4.2f'%((fill_step/maxsteps)*100),
                      'step=',step,'fill=',fill_step,'remain=',remain_step,'idx=',idx,'size=',bucket_size)    
    
            bucket_idx = int(step - idx) 
            if bucket_idx < fill_step-idx:
                break   
        nx = 2*mesh.Nx[0]
        ny = 2*mesh.Ny[0]
        xl,yl,low_stress = stresstensor(poisson, pos[0:nlow_nodes,:],u[bucket_idx,0:nlow_nodes],
                                     v[bucket_idx,0:nlow_nodes], nx, ny, comp, coord='deformed')
        nx = 2*mesh.Nx[1]
        ny = 2*mesh.Ny[1]
        xu,yu,up_stress = stresstensor(poisson,pos[nlow_nodes:-1,:],u[bucket_idx,nlow_nodes:-1],
                                     v[bucket_idx,nlow_nodes:-1], nx, ny, comp,coord='deformed')
        # extracting interface bonds corresponding to the time step
        while True:
            try:
                inter_data = pickle.load(inter_file)
            except:
                print('Interface file reading completed')       
            if inter_data[0]==t[i]:
                break
          
        inter_bonds = inter_data[1]
        segs = np.zeros(shape = (len(inter_bonds), 2,2))
        new_pos = pos + np.column_stack((u[bucket_idx,:],v[bucket_idx,:]))
        # create a line segments
        for j, [id1, id2] in enumerate(inter_bonds):
            x = (new_pos[id1,0], new_pos[id2,0])
            y = ([new_pos[id1,1], new_pos[id2,1]])
            segs[j,:,0] = x
            segs[j,:,1] = y

        line_collection = LineCollection(segs, linewidths = 1  , color = 'tab:red')

        # visualization
        fig, ax = plt.subplots(figsize = (10,10))
        ax_divider= make_axes_locatable(ax)
        # add an axes to the right of the main axes
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        if cbar is None:
            ul = max(np.nanmax(low_stress),np.nanmax(up_stress))
            ll = min(np.nanmin(low_stress),np.nanmin(up_stress))
            cbarlim = [ll, ul] 
        else:
            cbarlim = cbar    
        ax.set_title(r'$t = {%s}$'%t[i], fontsize = 16)
        ax.set_xlim([mesh.domain[0,0]-2, mesh.domain[0,1]+2])
        ax.set_ylim([mesh.domain[0,2]+10, mesh.domain[1,3]+2])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')
        # ax.yaxis.set_tick_params(labelbottom=False)
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.set_xticks([])
        # ax.set_yticks([])

        surf = ax.pcolormesh(xl,yl, low_stress, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1] )
        surf = ax.pcolormesh(xu, yu, up_stress, cmap = 'jet',
                  shading='auto',vmin=cbarlim[0], vmax=cbarlim[1])
        ax.add_collection(line_collection)
        fig.colorbar(surf, cax=cax, ticks = np.round(np.linspace(cbarlim[0], cbarlim[1],4),3))
        fig.savefig(sub_dir+f"{int(step)}", bbox_inches = 'tight', dpi = 300)
        plt.close()  
        with open(sub_dir + f'{str(t[i])}', 'wb') as plotdata:
            pickle.dump([xl,yl,xu,yu,low_stress,up_stress,segs, mesh.domain],plotdata)              
   
    disp_file.close()
    inter_file.close()

"""Spatial-temporal distribution of bond stretch"""
def spatial_time_stretch(extract_data):
    # load mesh object
    mesh = fc.load_mesh(extract_data)
    mesh.folder = extract_data

    # load displacement fields
    disp_file = h5py.File(mesh.folder + '/disp.h5', 'r')
    
    start = time.time()
    maxsteps = disp_file['u'][:,0].shape[0]
    bucket = 0
    fill_step = 0
    remain_step = maxsteps - fill_step
    bucket_size = 10000
    if maxsteps < bucket_size:
        bucket_size = maxsteps
    
    # load contact information
    top_nodes_ids = mesh.contact.surfaces[1]

    # get x position of top surface
    x = mesh.pos[top_nodes_ids,0]
    
    f = open(mesh.folder + '/contacts','rb')

    t = np.zeros(maxsteps)
    stretch = np.zeros(shape = (maxsteps,len(top_nodes_ids)))

    for step in range(maxsteps):
        # read displacement data in buckets
        if step == fill_step:
            u = disp_file['u'][fill_step:fill_step+bucket_size]
            v = disp_file['v'][fill_step:fill_step+bucket_size]
            bucket += 1
            fill_step += bucket_size
            remain_step = maxsteps - fill_step
            if remain_step < bucket_size:
                bucket_size = remain_step
            bucket_idx = 0
        try:
            item = pickle.load(f)
        except:
            print('file reading completed')   
            break

        t[step] = item[0]
        contacts = item[1]

        for id, neigh in contacts:
            id_disp = np.array([u[bucket_idx,id],v[bucket_idx,id]])
            neigh_disp = np.array([u[bucket_idx, neigh],v[bucket_idx, neigh]])
            rf = mesh.pos[neigh] + neigh_disp - mesh.pos[id] - id_disp
            lf = np.linalg.norm(rf,2)
            stretch[step,neigh-top_nodes_ids[0]] = lf
        bucket_idx += 1    
                  
    f.close()   
    disp_file.close()
    end = time.time()
    print('Time elpased: ', end-start)
    return t,x,stretch 



