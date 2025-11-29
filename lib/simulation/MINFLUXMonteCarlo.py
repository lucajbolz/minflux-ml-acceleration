import os, multiprocessing
script_name = os.path.basename(__file__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, dual_annealing, differential_evolution,shgo,linear_sum_assignment
import numpy as np
import pandas as pd
import jax as jax
import jax.numpy as jnp
from functools import partial
import ast
import copy

from lib.data_handling.data_analysis import Estimators
from lib.data_handling.mf_parser import MinfluxAnalysis, MinfluxPostProcessing
import lib.utilities as ut
from lib.constants import LAMBDA
from lib.plotting.style import Style
from lib.plotting.artefacts import Artefacts, Figures


#mpl.use('TkAgg')

def mod_fit(labels, vals, **kwargs):
    try:
        keys = kwargs['keys']
        results_df = pd.DataFrame(MinfluxAnalysis().fit_chunk(vals,**kwargs)).T.reset_index(drop=True)
        labels_df = pd.DataFrame([labels], columns=keys).reset_index(drop=True)
        new_rows = pd.concat([results_df, labels_df], axis=1)
        new_rows = new_rows.assign(**ast.literal_eval(vals.loc[:,'params'].iloc[0]))
        return new_rows
    except Exception as e:
        # Print or handle the exception here
        print(f"Exception in mod_fit: {e}")
        return pd.DataFrame()
    
def MLE(group_labels, group_df, **kwargs):
    try:
        keys = kwargs['keys']
        results_df = pd.DataFrame(MCAnalysis().getMLE(group_df,solver=kwargs['solver'])).T.reset_index(drop=True)
        
        #results_df = pd.DataFrame(MinfluxAnalysis().fit_chunk(group_df,**kwargs)).T.reset_index(drop=True)
        
        labels_df = pd.DataFrame([group_labels], columns=keys).reset_index(drop=True)
        new_rows = pd.concat([results_df, labels_df], axis=1)
        new_rows = new_rows.assign(**ast.literal_eval(group_df.loc[:,'params'].iloc[0]))
        return new_rows
    except Exception as e:
        # Print or handle the exception here
        print(f"Exception in MLE: {e}")
        return pd.DataFrame()
    
def process_group(group):
    try:
        group['d_norm'] = group.groupby(['chunk_size','bin_size', 'projection_id', 'chunk_id'])['d'].transform(lambda x: np.linalg.norm(x))
    except:
        pass
    return group

def pairwise_distances_jax(positions):
    squared_distances = jnp.sum((positions[:, jnp.newaxis, :] - positions[jnp.newaxis, :, :]) ** 2, axis=-1)
    distances = jnp.sqrt(squared_distances+1E-9)
    return distances

class MCAnalysis:

    def __init__(self):
        pass

    def model(self, params, chunk):
        """
        Model for a single chunk, i.e. N=1, 
        """
        # params is the set of optimization parameters, e.g. molecule positions
        model = Estimators().harmonic_model
        param_dict = ast.literal_eval(chunk.loc[:,'params'].iloc[0])
        param_dict['N0'] = 1
        predicted_model_params, _ = MCSimulation()._generate_params(param_dict, systems=params)

        phases = chunk['pos'].to_numpy()*4*np.pi/LAMBDA
        phases = phases.reshape((1,param_dict['P0'],-1,3))
        
        num_additional_dims = len(phases.shape) - (len(predicted_model_params.shape)-1)#minus one because of the three parameters stored as first dimension
        predicted_model_params = predicted_model_params[(...,) + (None,) * num_additional_dims]
        predicted_model_means = model(phases, predicted_model_params)
        return predicted_model_means#shape=(N=1,P,K,#pos/tuple)
    
    def Loglikelihood(self, model, params, measured_model_vals):
        """
        params to be optimized
        model underlying the measurement
        fixed_model_params specifies how was measured
        measured_model_vals , i.e. counts
        """
        predicted_model_means = model(params)+1E-3
        LogLikelihood = jnp.sum(measured_model_vals*jnp.log(predicted_model_means)-predicted_model_means)
        return LogLikelihood
    
    def getMLE(self, input_df, solver='SLSQP'):
        chunk = input_df.copy()        
        model = lambda p: self.model(p, chunk)
        param_dict = ast.literal_eval(chunk.loc[:,'params'].iloc[0])
        measured_model_vals = chunk['photons'].to_numpy().reshape((1,param_dict['P0'],-1,3))

        objective = lambda p: (-1)*self.Loglikelihood(model, p, measured_model_vals)
        jacobian = lambda p: jax.jacfwd(objective)(p)
        hessian = lambda p: jax.hessian(objective)(p)

        initPos = MCSimulation()._generate_points(1.5*param_dict['d0'], param_dict['M0'], 1, config=param_dict['config'])#param_dict['config'])
        lb = 2*param_dict['M0']*[-param_dict['d0']]
        ub = 2*param_dict['M0']*[param_dict['d0']]
        bounds = ut.BaseFunc().to_tuple((np.vstack([lb,ub]).T).tolist())

        # constrained optimization: single parameter for regular shapes.
        def constrained_objective(d):
            points = MCSimulation()._generate_points(d*param_dict['d0'], param_dict['M0'], 1, config=param_dict['config'])
            return objective(points)

        #crb = self.getCramerRaoBound(model, initPos)

        """fig, ax = plt.subplots()
        x, y = jnp.linspace(0,3,20), []
        for s in zip(x):
            a = s * initPos.reshape((param_dict['M0'],2))
            y.append(objective(a))
        y = jnp.hstack(y)

        # Create a contour plot with masked values
        ax.plot(x,y)
        plt.show()

        fig, ax = plt.subplots(1,2)
        x,y = jnp.linspace(-.5,.5,20), jnp.linspace(-.5,.5,20)
        X,Y = jnp.meshgrid(x,y)
        z1, z2 = [], []
        for p in zip(X.flatten(),Y.flatten()):
            a = jnp.array(p) + initPos.reshape((param_dict['M0'],2))
            z1.append(objective(a))
            z2.append(1)#jacobian(a)@jacobian(a).T)
        z1, z2 = jnp.hstack(z1), jnp.hstack(z2)
        Z1, Z2 = z1.reshape(X.shape), z2.reshape(X.shape)
        Z1_masked, Z2_masked = np.ma.masked_invalid(Z1), np.ma.masked_invalid(Z2)

        # Create a contour plot with masked values
        ax[0].pcolormesh(X, Y, Z1_masked, shading='auto')
        ax[1].pcolormesh(X, Y, Z2_masked, shading='auto')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Contour Plot with Masked Values')
        plt.show()"""


        
        f = constrained_objective#objective
        init = .3#initPos
        if solver=='SLSQP':
            options = {'ftol': 1e-5, 'eps': 1e-3}
            sol = minimize(f, init, method='SLSQP',options=options)
        if solver=='BFGS':
            #options = {'ftol': 1e-5, 'eps': 1e-3}
            sol = minimize(f, init, method='BFGS',jac=jacobian)
        if solver=='nelder-mead':
            sol = minimize(f, init, method='nelder-mead')
        if solver=='powell':
            sol = minimize(f, init, method='powell')
        if solver=='CG':
            sol = minimize(f, init, method='CG')
        if solver=='trust-ncg':
            sol = minimize(f, init, method='trust-ncg',jac=jacobian,hess=hessian)
        if solver=='trust-krylov':
            sol = minimize(f, init, method='trust-krylov',jac=jacobian, hess=hessian)
        if solver=='dual-annealing':
            sol = dual_annealing(f,bounds)
        if solver=='differential-evolution':
            sol = differential_evolution(f,bounds)
        if solver=='SHGO':
            cons = 1
            sol = shgo(f, bounds, constraints=cons)

        
        foundPos = MCSimulation()._generate_points(sol.x*param_dict['d0'], param_dict['M0'], 1, config=param_dict['config'])
        initPos = MCSimulation()._generate_points(init*param_dict['d0'], param_dict['M0'], 1, config=param_dict['config'])
        predicted_model_vals = model(foundPos)
        chi2 = np.sqrt(np.average((measured_model_vals-predicted_model_vals)**2/predicted_model_vals**2))

        chunk = pd.Series({'sol':list(foundPos)
                           ,'ground_truth':input_df.loc[:,'ground_truth'].iloc[0]
                           ,'init':initPos
                           ,'chi2':chi2
                           ,'success':sol.success
                           ,'N_fit':np.sum(measured_model_vals)
                           ,'N_avg': np.nanmean(measured_model_vals)
                           ,'time':chunk['time'].mean()
                           }
                           )

        return chunk
        
        if np.mean(photons)>3:
            chi2 = np.sqrt(np.average((photons-fit_vals)**2/fit_vals**2))
            chunk = pd.Series({'chi2':chi2,'a0':sol[0],'a1':sol[1],'x0':sol[2],'success':success,'N_fit':np.sum(photons),'N_avg': np.nanmean(photons),'time':chunk_df['time'].mean()})
        else:
            chunk = pd.Series({'chi2':0.1,'a0':1.,'a1':1.,'x0':0.,'success':True,'N_fit':np.sum(photons),'N_avg': np.nanmean(photons),'time':chunk_df['time'].mean()})
        return chunk
    
    def getFisherMatrix(self,model,params):
        J = jax.jacfwd(model)(params) # jacobi matrix
        val = jnp.diag(1/model(params)) # diagonal matrix of inverse model
        mask = J!=jnp.inf
        new_mask = mask[:,0] + mask[:,1]
        red_J = J[new_mask]
        red_val = val[new_mask.T].T[new_mask]
        FIM = red_J.T@red_val@red_J
        return FIM

    def getInverseFisherMatrix(self,model,params):
        """
        Returns inverse of Fisher Information Matrix.
        :param X: array of molecules positions
        """
        FIM = self.getFisherMatrix(model,params)
        if jnp.linalg.det(FIM) == 0:
            return np.zeros(FIM.shape)#TODO: return array of correct shape?
        else:
            return jnp.linalg.inv(FIM)

    def getCramerRaoBound(self,model,params):
        """
        Calculates the average CRB for a given constellation of molecules.
        :param X: array of molecule positions, shape=(M,dim)
        :param pattern: pattern object
        :return: sigma CRB [nm]
        """
        try:
            invFIM = self.getInverseFisherMatrix(model,params)
            trace = jnp.trace(invFIM)
            var, eig_vec = jnp.linalg.eig(invFIM)#jnp.linalg.eig(self._calc_invFIM(X,pattern))
            avg_sig = jnp.sqrt(trace / jnp.sqrt(invFIM.size))#invFIM should be quadratic
        except:
            avg_sig = np.nan
        CRB_dict = {'avg_sig': avg_sig}#, 'eig_sig': np.sqrt(var.real), 'eig_vec': eig_vec.real}
        return CRB_dict


class MultifluxPostProcessing:
    def __init__(self):
        pass

    def post_process(self,input_dir,output_dir,visualize=True):
        os.makedirs(output_dir+'post_processed/', exist_ok=True)
        results_file = ut.BaseFunc().find_files(input_dir+'processed/', lambda file: ut.BaseFunc().match_pattern(file, '.pkl', match='partial') & ('fitting-results' in file), max_files = 1)[0]
        results_df = pd.read_pickle(results_file)
        
        artefacts = Artefacts()
        try:
            #----------------------
            # initial setp of dataframe
            results = results_df.reset_index()                
            available_targets = results['chunk_size'].unique()
            print(f"Available targets: {available_targets}")
            original_data = results.copy()
            collected_results = pd.DataFrame({})

            results = original_data.copy()
            #----------------------------
            # filter out failed fits (chunk-wise) via residuals
            key_columns = ['file', 'chunk_size', 'bin_size', 'chunk_id']
            mask = ((results['chi2'] > np.inf*.55) | (results['chi2'] < 0.))# Create a boolean mask to identify rows to remove
            combinations_to_remove = results.loc[mask, key_columns].drop_duplicates()# Get the key columns as tuples for combinations to remove
            filtered_results = results.merge(combinations_to_remove, on=key_columns, how='left', indicator=True)# Filter the DataFrame using merge
            filtered_results = filtered_results[filtered_results['_merge'] == 'left_only'].drop(columns='_merge')

            res = filtered_results.reset_index(drop=True).copy()
            for i, row in res.iterrows():
                TruePositions = np.array(row.ground_truth).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)
                FoundPositions = np.array(row.sol).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)
                InitialPositions = np.array(row.init).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)

                C = cdist(TruePositions[0], FoundPositions[0])
                row_idx, col_idx = linear_sum_assignment(C)#_, assigment = linear_sum_assignment(C)
                PermutatedPositions = FoundPositions[0][row_idx][col_idx]#FoundPositions[0][assigment]
                ds = np.linalg.norm(PermutatedPositions-TruePositions[0],axis=-1)
                rmse = np.sqrt(np.mean(ds**2))
                res.loc[i,'rmse'] = rmse
                #res.loc[i,'perm'] = list(PermutatedPositions.flatten())
            
            res.to_pickle(output_dir+'post_processed/' + ut.Labeler().stamp('all-postprocessing-results')[0] + '.pkl') 
            
            for target in available_targets:
                print(f"Chosen target: {target}")
                os.makedirs(output_dir+f'post_processed/{int(target)}/', exist_ok=True)
                results = original_data.copy()
                #results = results.loc[(results['chunk_size']==500)]
                results = results.loc[(results['chunk_size']==target)]#

                # for each file compare solution and groundtruth via permutations
                colors = ['r', 'orange', 'b', 'c', 'm', 'y', 'k', 'g', 'purple', 'brown']
                markers = ['.', 'o', 's', '^', 'D', 'v', '>', 'x', '+', '*']
                
            
                for i, row in res.iterrows():
                    TruePositions = np.array(row.ground_truth).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)
                    FoundPositions = np.array(row.sol).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)
                    InitialPositions = np.array(row.init).reshape((1,row.M0,2)) * LAMBDA/(4*np.pi)

                    C = cdist(TruePositions[0], FoundPositions[0])
                    row_idx, col_idx = linear_sum_assignment(C)#_, assigment = linear_sum_assignment(C)
                    PermutatedPositions = FoundPositions[0][row_idx][col_idx]#FoundPositions[0][assigment]
                    ds = np.linalg.norm(PermutatedPositions-TruePositions[0],axis=-1)

                    fig, ax = plt.subplots()
                    ax.scatter(TruePositions.T[0],TruePositions.T[1],c='g',label='Truth')
                    for p in range(row.M0):
                        plt.plot([TruePositions[0,p,0], PermutatedPositions[p,0]], [TruePositions[0,p,1], PermutatedPositions[p,1]], c=colors[i],alpha=.4)
                    ax.scatter(FoundPositions[0].T[0],FoundPositions[0].T[1],c=colors[i],marker=markers[i],alpha=.4,label=f'{row.solver} rmse: {round(rmse,1)}')
                    ax.scatter(InitialPositions.T[0],InitialPositions.T[1],c='g',marker='x',label='Init')
                    circle = plt.Circle((0,0), row.d0, color='blue', fill=False)
                    ax.add_patch(circle)
                    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    legend.get_frame().set_alpha(0.1)
                    ax.set_xlim(-2*row.d0,2*row.d0)
                    ax.set_ylim(-2*row.d0,2*row.d0)
                    ax.set_aspect('equal')
                    Figures().save_fig(fig, f'res{i}',meta={'generating script': script_name}, out_path=output_dir+f'post_processed/{int(target)}/')


            if False:
                # First create the x and y coordinates of the points.
                n_angles = 20
                n_radii = 20
                min_radius = 1
                max_radius = 40
                radii = np.linspace(min_radius, max_radius, n_radii)
                angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
                
                x, y, z1, z2 = [], [], [], []
                init_norm = initPos/np.linalg.norm(initPos,axis=-1)[:,:,np.newaxis]
                for r in radii:
                    for a in angles:
                        scaledPos = r*init_norm
                        rotation_matrix = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
                        # Rotate all vectors in the matrix
                        rotatedPos = np.dot(scaledPos, rotation_matrix)
                        p = rotatedPos.flatten()*4*jnp.pi/LAMBDA
                        x.append(r*np.cos(a))
                        y.append(r*np.sin(a))
                        z1.append(objective(p))
                        z2.append(jnp.dot(jacobian(p),jacobian(p)))

                z1, z2 = np.hstack(z1), np.hstack(z2)
                triang = tri.Triangulation(x, y)

                refiner = tri.UniformTriRefiner(triang)
                tri_refi1, z_test_refi1 = refiner.refine_field(z1, subdiv=3)
                tri_refi2, z_test_refi2 = refiner.refine_field(z2, subdiv=3)

                fig, ax = plt.subplots(1,2)
                ax[0].set_aspect('equal')
                ax[0].triplot(triang, lw=0.5, color='white')
                d = np.abs(jnp.min(z1)-jnp.max(z1))
                levels = np.linspace(jnp.min(z1)-.01*d, jnp.max(z1)+.01*d, 10,endpoint=True)
                ax[0].tricontourf(tri_refi1, z_test_refi1, levels=levels, cmap='terrain',label='Likelihood')
                ax[0].tricontour(tri_refi1, z_test_refi1, levels=levels,
                            colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                            linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
                ax[0].scatter(TruePositions[0].T[0],TruePositions[0].T[1],c='r',marker='x',label='Truth')
                ax[0].legend()

                ax[1].set_aspect('equal')
                ax[1].triplot(triang, lw=0.5, color='white')
                d = np.abs(jnp.min(z2)-jnp.max(z2))
                levels = np.linspace(jnp.min(z2)-.01*d, jnp.max(z2)+.01*d, 10,endpoint=True)
                ax[1].tricontourf(tri_refi2, z_test_refi2, levels=levels, cmap='terrain',label='Jacobian')
                ax[1].tricontour(tri_refi2, z_test_refi2, levels=levels,
                            colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                            linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
                ax[1].scatter(TruePositions[0].T[0],TruePositions[0].T[1],c='r',marker='x',label='Truth')
                ax[1].legend()
                plt.show()     
        except:
            pass
        pass

class MCSimulation:

    def __init__(self):
        self.rng = np.random.default_rng()
        self.model = Estimators().harmonic_model

    def run(self, params, bootstrap_dicts):
        new_dir = self.generate_data(params)
        new_dir = self.analyze(new_dir, bootstrap_dicts)

        postprocessor = MultifluxPostProcessing()
        postprocessor.post_process(new_dir,new_dir,visualize=False)
        return new_dir

    def generate_data(self, param_dicts):
        """
        MC simulation of last MINFLUX iteration
        """
        dir_str, todays_dir = ut.Labeler().stamp('MULTIFLUX-session')
        new_dir = todays_dir + dir_str + '/'
        
        for idx,params in enumerate(param_dicts):
            os.makedirs(os.path.join(new_dir,f'data/batch{idx}/'), exist_ok=True)
            df = pd.DataFrame({})
            # get simulation parameters
            N = params.get('N0',5)#number of systems per config
            M = params.get('M0',3)#number of molecules per system
            P = params.get('P0',M**2)#number of projections
            K = params.get('K0',100)#number of tuples to simulate in each segment
            L = params.get('L0',70)# MINFLUX L in nm

            # physical parameters to calculate model parameters:
            # 2D distance, brightness, relative brightness, initial fringe contrast, background rate
            model_params, generated_positions = self._generate_params(params)
            
            ground_truth = generated_positions.reshape((N,M,2))
            ground_truth = np.array([ground_truth[n].flatten() for n in range(N)])[:, np.newaxis, np.newaxis, np.newaxis,:]
            ground_truth = np.tile(ground_truth, (1, P, K, 3,1))

            # get random phases as sampling points
            # #N systems, #projections=M**2, K Tuples per projection, points per tuple, 
            positions = 10*(self.rng.random((N,P,K,3))-.5)+np.array([-L/2,0,L/2])[np.newaxis,np.newaxis,np.newaxis,:]
            phases = 4*np.pi/LAMBDA * positions

            tuple_index = np.arange(K)[np.newaxis, np.newaxis, :, np.newaxis]
            tuple_index = np.tile(tuple_index, (N, P, 1, 3))

            file_index = np.array([ut.Labeler().id_generator(size=10) for i in range(N)])[:, np.newaxis, np.newaxis, np.newaxis]
            file_index = np.tile(file_index, (1, P, K, 3))

            projection_index = np.arange(P)[np.newaxis, :, np.newaxis, np.newaxis]
            projection_index = np.tile(projection_index, (N, 1, K, 3))

            # adjust dimensions and get model values by broadcasting
            num_additional_dims = len(phases.shape) - (len(model_params.shape)-1)#minus one because of the three parameters stored as first dimension
            model_params = model_params[(...,) + (None,) * num_additional_dims]
            model_means = self.model(phases, model_params)
            model_means = np.nan_to_num(model_means, nan=0)

            # get noisy photon numbers
            photons = self.rng.poisson(model_means)

            # create dataframe, automatic labelling of segments (do not need to filter)
            dictionary = dict(
                pos=positions.flatten()
                ,photons = photons.flatten()
                ,tuple=tuple_index.flatten()
                ,projection_id=projection_index.flatten()
                ,file=file_index.flatten()
                ,ground_truth = list(ground_truth.reshape(-1, M*2))
                )
            new_df = pd.DataFrame(dictionary)
            new_df['params'] = str(params)
            df = pd.concat([df, new_df], ignore_index=False)
            df.to_pickle(new_dir + f'data/batch{idx}/' + 'data.pkl')
        return new_dir
    
    def analyze(self, dir, bootstrap_dicts):
        files = ut.BaseFunc().find_files(os.path.join(dir,'data'), lambda file: 'data.pkl' in file, max_files=np.inf) # load pkl
        for idx,batch in enumerate(files):
            try:
                os.makedirs(os.path.join(dir,f'processed/batch{idx}/'), exist_ok=True)
                df = pd.read_pickle(batch)
                results = pd.DataFrame({})
                for i,i_dict in enumerate(bootstrap_dicts):
                    try:
                        i_dict = i_dict|dict(output=dir)
                        chunked_df = df.groupby(['file'],group_keys=False).apply(MinfluxAnalysis().assign_chunk_id, **i_dict).copy()
                        keys = ['file','chunk_id','bin_id']
                        groups = chunked_df.groupby(keys,group_keys=False)
                        kwargs = i_dict|dict(keys=keys)
                        task = partial(MLE,**kwargs)
                        processed_groups = []
                        #for group in groups:
                        #    processed_groups.append(task(*group))
                        num_cores = multiprocessing.cpu_count()
                        with multiprocessing.Pool(processes=num_cores) as pool:
                            processed_groups = pool.starmap(task, groups)
                        new_rows = pd.concat(processed_groups).reset_index(drop=True)
                        new_rows = new_rows.assign(**i_dict)
                        results = pd.concat([results, new_rows], ignore_index=True)
                    except:
                        continue
                        raise Exception('Analysis failed!')
                valid_results = results.copy()#if there is any valid chunk, keep the trace
                valid_results = valid_results.loc[valid_results['success']==True]
                valid_results.to_pickle(os.path.join(dir,f'processed/batch{idx}/')+'local-results.pkl')
            except:
                print(f'Analysis of batch {idx} failed')
        # collect all local results
        files = ut.BaseFunc().find_files(os.path.join(dir,'processed'), lambda file: ut.BaseFunc().match_pattern(file, 'local-results', match='partial'),max_files=np.inf)
        results = pd.DataFrame({})
        for file in files:
            df = pd.read_pickle(file)
            results = pd.concat([results, df], ignore_index=True)
        results.to_pickle(os.path.join(dir,f'processed/')+'fitting-results.pkl')
        return dir
    
    def _generate_params(self,param_dict,systems=None):
        """
        Generate an array of configs for the simulations.

        :param param_dict: dict
            Dictionary containing parameters for simulations.
            - 'M0' (int): Number of molecules per system (default: 3).
            - 'P0' (int): Parameter P (default: M^2, where M is 'M0').
            - 'I0' (int): Parameter I (default: 1).
            - 'L0' (int): MINFLUX L in nm (default: 70).
            - 'N0' (int): Number of systems per config (default: 5).
            - 'K0' (int): Number of tuples to simulate in each segment (default: 100).
            - 'd0' (int): Scale parameter of the system in nm (default: 30).
            - 'gamma0' (int): Average molecule brightness (default: 13).
            - 'r0' (int): Relative brightness (default: 0).
            - 'alpha0' (int): Alpha parameter (default: 1).
            - 'sig_alpha0' (int): Sigma alpha parameter (default: 0).
            - 'beta0' (int): Beta parameter (default: 0).
        :param systems: numpy.ndarray, optional
            Array representing system configurations.

        :return: tuple
            - params0 (numpy.ndarray): Array containing parameters a0, a1, and phi in (3, axis * segments * N) shape.
            - systems (numpy.ndarray): Updated system configurations.
            - phis (numpy.ndarray): Array of phase values.

        Generates parameters a0, a1, and phi for simulations in (3, axis * segments * N) shape.
        The order of segments is two-molecule segment, single-molecule segment, and background.

        Notes:
        - Systems need to be in units of phase and flattened array.
        - TODO: Add multiplication by a suitable gamma matrix that incorporates all brightnesses.
        - TODO: Update global I0 in a1 calculation.
        - Phase offset is assumed to be zero since we're centered on the effective COM.
        """
        config = param_dict.get('config','random')
        M = param_dict.get('M0',3)#number of molecules per system
        P = param_dict.get('P0',M**2)
        I = param_dict.get('I0',1)
        L = param_dict.get('L0',70)# MINFLUX L in nm
        N = param_dict.get('N0',5)#number of systems per config
        K = param_dict.get('K0',100)#number of tuples to simulate in each segment
        d = param_dict.get('d0',30)#scale parameter of system in nm
        gamma = param_dict.get('gamma0',13)#average molecule brightness
        r = param_dict.get('r0',0)#relative brightness
        alpha = param_dict.get('alpha0',1)
        sig_alpha = param_dict.get('sig_alpha0',0)
        beta = param_dict.get('beta0',0)
        
        if systems is None:
            systems = self._generate_points(d, M, N, config=config)
        else:
            assert 1==1#TODO: assert that systems has proper shape and type
        systems = systems.reshape((N,M,2))

        # systems need to be in units of phase!
        phis = jnp.linspace(0, jnp.pi/2, P, endpoint=True)
        cos_phis = jnp.cos(phis)
        sin_phis = jnp.sin(phis)

        # Generate projections
        cos_phis_sin_phis = jnp.stack([cos_phis, sin_phis], axis=1)
        projections = cos_phis_sin_phis[jnp.newaxis, :, jnp.newaxis,:] * systems[:, jnp.newaxis, :, :]

        # Calculate pairwise distances for projections
        pairwise_distances = jnp.zeros((N, P, M, M))
        for n in range(N):
            for k in range(P):
                subsystem_positions = projections[n, k]
                reshaped_positions = subsystem_positions.reshape(-1, 2)
                subsystem_pairwise_distances = pairwise_distances_jax(reshaped_positions)
                pairwise_distances = pairwise_distances.at[n,k].set(subsystem_pairwise_distances)

        C = jnp.cos(pairwise_distances)#element-wise cosine of pairwise distances matrix
        #TODO: multiply by suitable gamma matrix, that incorporates all brightnesses
        g = jnp.ones(M)# single molecule relative brightness self.rng.uniform(0,1,M)
        # Construct the matrix G from the vector x
        G = jnp.outer(g, g)
        matrix = jnp.matmul(G,C)#matrix multiplication
        trace = jnp.trace(matrix, axis1=-2, axis2=-1)
        Gammas = jnp.sqrt(trace+1E-9)#shape(N systems, # projections)

        deltas = jnp.linalg.norm(projections,axis=-1)# shape N,P,M
        deltaks = jnp.arctan(jnp.dot(deltas,g))
        
        a1, a0, phi0 = np.empty((3,N,P))#shape = (#parameters, N systems, K tuples, #projections)
        alphar = self.rng.normal(loc=alpha, scale=sig_alpha, size=N)#generate random alphas for each system
        alphar = np.where(alphar > 1, 1, alphar)
        alphat = np.tile(alphar[:,np.newaxis],(1,P))
        
        a1 = 2 * alphat * Gammas#TODO: global I0!!!
        a0 = beta + (1+alphat**2)*M*np.mean(g)
        phi0 = deltaks# phase offset assumed to be zero since we're centered on effective COM

        params0 = jnp.concatenate(jnp.array([[I*a0],[I*a1],[phi0]]),axis=0)#shape=(#parameters, N systems, #projections)
        return params0, systems.flatten()
    
    def _generate_points(self, lateral_scale, num_molecules, num_systems, config='random'):
        """
        Generate positions in units of phase.

        :param lateral_scale: float
            Scale parameter for lateral positions.
        :param num_molecules: int
            Number of molecules.
        :param num_systems: int
            Number of systems.
        :param mode: str, optional
            Mode for position generation: 'random' or 'polygon' (default: 'random').

        :return: numpy.ndarray
            Generated positions in units of phase.

        Generates positions in units of phase based on the specified parameters and mode.
        If mode is 'random', positions are randomly distributed within the specified lateral scale.
        If mode is 'polygon', positions are distributed along a polygon within the lateral scale.

        .. note::
            positions are scaled by 4 * np.pi / LAMBDA.
        """
        assert config in ['random','line','grid','polygon'], 'Invalid configuration for point generation.'
        if config=='random':
            x, y = (num_molecules-1) * lateral_scale * self.rng.uniform(-.5,.5,(2,num_systems,num_molecules))
        if config=='polygon':
            angles = np.linspace(0, 2 * np.pi, num_molecules, endpoint=False) + 0*np.pi*(self.rng.uniform(0,1,(num_systems,1))-0.5)
            x = lateral_scale/2 * np.cos(angles)
            y = lateral_scale/2 * np.sin(angles)
        if config=='line':
            x = (num_molecules-1) * lateral_scale * np.linspace(-.5, .5, num_molecules, endpoint=True)
            x = np.tile(x[np.newaxis,:],(num_systems,1))
            y = 0*x
        if config == 'grid':
            # Calculate closest square grid
            closest_square = int(np.ceil(np.sqrt(num_molecules))) ** 2
            rows = int(np.sqrt(closest_square))
            cols = rows
            
            # Generate positions in a grid
            x = (rows-1) * lateral_scale * np.linspace(-.5, .5, cols)
            y = (cols-1) * lateral_scale * np.linspace(-.5, .5, rows)
            x, y = np.meshgrid(x, y)
            x = x.flatten()[:num_molecules]
            x = np.tile(x[np.newaxis,:],(num_systems,1))
            y = y.flatten()[:num_molecules]
            y = np.tile(y[np.newaxis,:],(num_systems,1))
        #x -= np.mean(x)
        #y -= np.mean(y)
        positions = np.concatenate([x[:,:,np.newaxis],y[:,:,np.newaxis]],axis=2) *4*jnp.pi/LAMBDA
        
        #fig, ax = plt.subplots()
        #ax.scatter(positions[0].T[0],positions[0].T[1])
        #circle = plt.Circle((0,0), lateral_scale/2, color='blue', fill=False,zorder=-1)
        #ax.add_patch(circle)
        #ax.set_aspect('equal')
        #plt.show()
        return positions.flatten()
    
        

if __name__=='__main__':
    p0 = dict(
            I0=20
            ,M0=8
            ,P0=4
            ,N0=2
            ,K0=300
            ,d0=30
            ,gamma0=60
            ,r0=0.0
            ,alpha0=1.
            ,sig_alpha0=.0
            ,beta0=0.1
            ,L0=30
            ,config='polygon'
            )
    d0 = dict(
        mode='photons'
        ,chunk_size=5000
        ,overlap=0.0
        ,bin_size=5000
        ,max_chunks=1
        ,plot=False
        ,output=None
        )
    d0 = d0|dict(solver='powell')
    #d0|dict(solver='SLSQP')
    #,d0|dict(solver='BFGS')
      
    #,d0|dict(solver='nelder-mead')
    #,d0|dict(solver='CG')
    #,d0|dict(solver='trust-ncg')
    #,d0|dict(solver='trust-krylov')
    #,d0|dict(solver='dual-annealing')
    #,d0|dict(solver='differential-evolution')
        
    MCSimulation().run(params=[p0],bootstrap_dicts=[d0])