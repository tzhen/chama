from __future__ import print_function

import chama
import chama.optimize
import chama.ray_cast as rc
import math
import pandas as pd
import numpy as np
import pprint

def tuple_to_str(tuple):
    items = list()
    for t in tuple:
        items.append(str(int(t)))
    return '-'.join(items)

def tuple_list_to_str_list(tuple_list):
    str_list = list()
    for t in tuple_list:
        str_list.append(tuple_to_str(t))
    return str_list

def empty_room():
    grid = rc.Grid(xu=100.0, yu=100.0, zu=5.0, nx=50, ny=50, nz=1)
    # print(grid._grid_open[49, 50, 0])

    tspace_120, hspace_0 = rc.get_camera_angle_spaces(theta_fov=120.0, horizon_fov=0.0, n_theta=300, n_horizon=1)
    tspace_90, hspace_0 = rc.get_camera_angle_spaces(theta_fov=90.0, horizon_fov=0.0, n_theta=300, n_horizon=1)
    tspace_60, hspace_0 = rc.get_camera_angle_spaces(theta_fov=60.0, horizon_fov=0.0, n_theta=300, n_horizon=1)

    cam1 = rc.get_camera_intersections(grid, x=1.0, y=75, z=2.5, theta_deg=0.0, theta_deg_space=tspace_120, 
                                    horizon_deg=0.0, horizon_deg_space=hspace_0, dist_step=0.1)
    cam1_str = tuple_list_to_str_list(cam1)
    cam1_ij_list = []
    for (i,j,k) in cam1:
        cam1_ij_list.append((i,j,'rgba(255, 128, 0, 0.7)'))

    cam2 = rc.get_camera_intersections(grid, x=1.0, y=50, z=2.5, theta_deg=0.0, theta_deg_space=tspace_90,
                                    horizon_deg=0.0, horizon_deg_space=hspace_0, dist_step=0.1)
    cam2_str = tuple_list_to_str_list(cam2)
    cam2_ij_list = []
    for (i,j,k) in cam2:
        cam2_ij_list.append((i,j,'rgba(0, 128, 0, 0.7)'))


    cam3 = rc.get_camera_intersections(grid, x=1.0, y=25, z=2.5, theta_deg=0.0, theta_deg_space=tspace_60,
                                    horizon_deg=0.0, horizon_deg_space=hspace_0, dist_step=0.1)
    cam3_str = tuple_list_to_str_list(cam3)
    cam3_ij_list = []
    for (i,j,k) in cam3:
        cam3_ij_list.append((i,j,'rgba(0, 0, 128, 0.7)'))

    cam3_ij_list_trunc = []
    for (i,j,k) in cam3:
        if i**2 + j**2 + k**2 < 1200:
            cam3_ij_list_trunc.append((i,j,'rgba(0, 0, 128, 0.7)'))

    # grid.plotly_2d_grid('cam1', ij_list=cam1_ij_list)
    # grid.plotly_2d_grid('cam2', ij_list=cam2_ij_list)
    # grid.plotly_2d_grid('cam3', ij_list=cam3_ij_list)
    # grid.plotly_2d_grid('cam1_cam2', ij_list=cam2_ij_list+cam1_ij_list)
    # grid.plotly_2d_grid('all_cams', ij_list=cam2_ij_list+cam1_ij_list+cam3_ij_list)
    # grid.plotly_2d_grid('all_cams_trunc', ij_list=cam2_ij_list+cam1_ij_list+cam3_ij_list_trunc)

    coverage_dict = {'Sensor':['cam1', 'cam2', 'cam3'], 'Coverage': [cam1_str, cam2_str, cam3_str]}
    coverage = pd.DataFrame(coverage_dict)

    entity_dict = {'Entity': [], 'Weight': [], 'Tuple': []}
    all_grids = grid.get_all_grid_ijk()
    # print(len(all_grids))
    for (i,j,k) in all_grids:
        grid_str = tuple_to_str((i,j,k))
        weight = 1.0
        if j > 45 and i < 10:
            weight = 5000.0
        entity_dict['Entity'].append(grid_str)
        entity_dict['Weight'].append(weight)
        entity_dict['Tuple'].append((i,j,k))
    entities = pd.DataFrame(entity_dict)

    cov_opt = chama.optimize.CoverageFormulation()
    results = cov_opt.solve(coverage=coverage, entity=entities, sensor_budget=1, redundancy=0)
    print(results['FractionDetected'], results['Sensors'])

    # test sensor cost
    sensor_dict = {'Sensor': ['cam1', 'cam2', 'cam3'], 'Cost': [5, 3, 1]}
    sensor = pd.DataFrame(sensor_dict)

    results = cov_opt.solve(coverage=coverage, entity=entities, sensor=sensor, sensor_budget=2, redundancy=0)
    print(results['FractionDetected'], results['Sensors'])
#    print(results['EntityAssessment'])

    results = cov_opt.solve(coverage=coverage, entity=entities, sensor=sensor, sensor_budget=2, redundancy=0, use_sensor_cost=True)
    print(results['FractionDetected'], results['Sensors'])

    # test priority grids
    results = cov_opt.solve(coverage=coverage, entity=entities, sensor_budget=1, redundancy=0, use_entity_weight=True)
    print(results['FractionDetected'], results['Sensors'])
#    print(results['EntityAssessment'])
#    print(results['Objective'])

    priority_ij_list = []
    for (idx, t) in enumerate(entity_dict['Tuple']):
        if entity_dict['Weight'][idx] > 10:
            priority_ij_list.append((t[0], t[1],'rgba(128, 0, 0, 0.7)'))
    grid.plotly_2d_grid('cam2_weight', ij_list=cam2_ij_list+priority_ij_list)
    grid.plotly_2d_grid('cam1_weight', ij_list=cam1_ij_list+priority_ij_list)

    quit()



    ij_list = []
    for (i,j,k) in cam2:
        ij_list.append((i,j,'rgba(128, 0, 0, 0.7)'))

    grid.plotly_2d_grid('cam2', ij_list=ij_list, range={'x': [0,100], 'y':[0,100]})

    ij_list = []
    for (i,j,k) in cam3:
        ij_list.append((i,j,'rgba(0, 0, 128, 0.7)'))

    grid.plotly_2d_grid('cam3', ij_list=ij_list, range={'x': [0,100], 'y':[0,100]})


def half_empty_room():
    ###
    # define the tank farm
    # BROKEN AS OF 3/23/18
    ###
    x_size = 10
    y_size = 10
    z_size = 5

    grid = rc.Grid(xu=float(x_size), yu=float(y_size), zu=float(z_size), nx=5, ny=5, nz=1)
    # print(grid._y, grid._x, grid._z)
    # define tanks at 30, and 70 and 110 with a "radius" of 10 blocks
    radius = 1
    height = 5
    xy = np.array([[2,2]])
    # a = np.arange(8,50,16)
    # xy = np.column_stack((a,a))
    # xy = np.array([[5,10],[12,10], [25,10], [25,20], [25,30],[40,30], [40,20], [40,10], [10,40]])

    for xc,yc in zip(xy[:,0],xy[:,1]):
        for xcd in range(-radius,radius+1):
            for ycd in range(-radius,radius+1):
                for zd in range(0,height+1):
                    grid.add_obstacle(xc+xcd, yc+ycd, zd)

    df_grid = pd.DataFrame(columns=['Grid','Open'])
    df_grid['Grid'] = grid._grid_open.keys()
    df_grid['Open'] = grid._grid_open.values()
    print(df_grid)

    camera_intersect = dict()

    theta_space, horizon_space = rc.get_camera_angle_spaces(theta_fov=60.0, horizon_fov=0.0, n_theta=100, n_horizon=1)
    
    x = range(1,x_size,2)
    y = range(1,y_size,2)
    z = 2.5

    # for xc in x:
    #     for yc in y:
    #         camera_intersect[(xc,yc,z)] = {}
    #         for ang in [0.0, 90., 180., 270.]:
    #             camera_intersect[(xc,yc,z)][ang] = \
    #                 rc.get_camera_intersections(grid=grid, x=xc, y=yc, z=z, theta_deg=ang, theta_deg_space=theta_space, horizon_deg=0.0, horizon_deg_space=horizon_space, dist_step=0.25)

    for xc in x:
        for yc in y:
            for ang in [0.0, 90., 180., 270.]:
                camera_intersect[(xc,yc,z,ang)] = \
                    rc.get_camera_intersections(grid=grid, x=xc, y=yc, z=z, theta_deg=ang, theta_deg_space=theta_space, horizon_deg=0.0, horizon_deg_space=horizon_space, dist_step=0.1)

    
    # df = pd.DataFrame(camera_intersect,columns=camera_intersect.keys()).T
    # set_locations = df.index
    # set_angles = df.columns

    # out_loc = []
    # out_dir = []
    # out_obs = []

    # for loc in set_locations:
    #     out_loc.append(loc[0:2])
    #     out_dir.append(loc[3])
    #     out_obs.append(tuple([int(i) for i in val]))

    # df2 = pd.DataFrame(columns=['Location','Direction','Observed'])
    # df2['Location'] = out_loc
    # df2['Direction'] = out_dir
    # df2['Observed'] = out_obs
    # print(df2)

    # ij_list = None
    # for (i,j,k) in df2['Observed']:
    #     ij_list.append((i-1,j-1,'green'))
    # grid.plotly_2d_grid('cam1',ij_list=ij_list)


    ## Output
    # with open('/Users/tzhen/repositories/workfiles/FireDetector/rc_data.py', 'w') as f:
    #     print('camera_intersect = ', camera_intersect, file=f)


    # df_grid.to_csv('~/repositories/workfiles/FireDetector/grid_data.csv') #export to csv the open grids
    # df2.to_csv('~/repositories/workfiles/FireDetector/test_ray_cast_data.csv') #export to csv df2


### Example 5x5

def ex_5x5():
    x_size = 10
    y_size = 10
    z_size = 10

    grid = rc.Grid(xu=float(x_size), yu=float(y_size), zu=float(z_size), nx=5, ny=5, nz=1)
    radius = 1
    height = 5
    xy = np.array([[2,2]]) #Obstacles

    for xc,yc in zip(xy[:,0],xy[:,1]):
        for xcd in range(-radius,radius+1):
            for ycd in range(-radius,radius+1):
                for zd in range(0,height+1):
                    grid.add_obstacle(xc+xcd, yc+ycd, zd)


    xy = np.array([[1,9,0.0],[9,1,90.0],[1,1,0.0],[1,3,90.0]]) # Solution [x,y,angle]
    z = 2.5

    camera_intersect = dict()
    theta_space, horizon_space = rc.get_camera_angle_spaces(theta_fov=60.0, horizon_fov=0.0, n_theta=100, n_horizon=1)

    for xc,yc,ang in zip(xy[:,0],xy[:,1],xy[:,2]):
        camera_intersect[(xc,yc,z)] = {}
        
        camera_intersect[(xc,yc,z)][ang] = \
            rc.get_camera_intersections(grid=grid, x=xc, y=yc, z=z, theta_deg=ang, theta_deg_space=theta_space, horizon_deg=0.0, horizon_deg_space=horizon_space, dist_step=0.1)

    df = pd.DataFrame(camera_intersect,columns=camera_intersect.keys()).T
    set_locations = df.index
    set_angles = df.columns

    out_loc = []
    out_dir = []
    out_obs = []

    for xc,yc,ang in zip(xy[:,0],xy[:,1],xy[:,2]):
        for val in camera_intersect[(xc,yc,z)][ang]:
            out_loc.append((xc,yc,z))
            out_dir.append(ang)
            out_obs.append(tuple([int(i) for i in val]))

    df2 = pd.DataFrame(columns=['Location','Direction','Observed'])
    df2['Location'] = out_loc
    df2['Direction'] = out_dir
    df2['Observed'] = out_obs
    # print(df2)

    ij_list = []
    for (i,j,k) in df2['Observed']:
        ij_list.append((i,j,'rgb(50,205,50)'))
    # print(ij_list)
    grid.plotly_2d_grid('cam1',ij_list=ij_list)


### Example 50x50
def ex_50x50():
    x_size = 100
    y_size = 100
    z_size = 10

    grid = rc.Grid(xu=float(x_size), yu=float(y_size), zu=float(z_size), nx=50, ny=50, nz=1)
    radius = 3
    height = 5
    xy = np.array([[5,10],[12,10], [25,10], [25,20], [25,30],[40,30], [40,20], [40,10], [10,40]]) #Obstacles

    for xc,yc in zip(xy[:,0],xy[:,1]):
        for xcd in range(-radius,radius+1):
            for ycd in range(-radius,radius+1):
                for zd in range(0,height+1):
                    grid.add_obstacle(xc+xcd, yc+ycd, zd)


    xy = np.array([[67,85,180.],[3,45,0.],[77,99,270.]]) # Solution [x,y,angle]
    z = 2.5

    camera_intersect = dict()
    theta_space, horizon_space = rc.get_camera_angle_spaces(theta_fov=60.0, horizon_fov=0.0, n_theta=100, n_horizon=1)

    for xc,yc,ang in zip(xy[:,0],xy[:,1],xy[:,2]):
        camera_intersect[(xc,yc,z)] = {}
        
        camera_intersect[(xc,yc,z)][ang] = \
            rc.get_camera_intersections(grid=grid, x=xc, y=yc, z=z, theta_deg=ang, theta_deg_space=theta_space, horizon_deg=0.0, horizon_deg_space=horizon_space, dist_step=0.1)

    df = pd.DataFrame(camera_intersect,columns=camera_intersect.keys()).T
    set_locations = df.index
    set_angles = df.columns

    out_loc = []
    out_dir = []
    out_obs = []

    for xc,yc,ang in zip(xy[:,0],xy[:,1],xy[:,2]):
        for val in camera_intersect[(xc,yc,z)][ang]:
            out_loc.append((xc,yc,z))
            out_dir.append(ang)
            out_obs.append(tuple([int(i) for i in val]))

    df2 = pd.DataFrame(columns=['Location','Direction','Observed'])
    df2['Location'] = out_loc
    df2['Direction'] = out_dir
    df2['Observed'] = out_obs
    # print(df2)

    ij_list = []
    for (i,j,k) in df2['Observed']:
        ij_list.append((i,j,'rgb(50,205,50)'))
    # print(ij_list)
    grid.plotly_2d_grid('cam1',ij_list=ij_list)


if __name__ == '__main__':
    empty_room()

