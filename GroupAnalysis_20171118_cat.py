
# coding: utf-8

# In[ ]:

from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.fsl.utils import Merge
from nipype.interfaces.fsl.model import Randomise, Cluster
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.freesurfer.model import Binarize

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

#other study-specific variables
#project_home = '/Volumes/iang/active/BABIES/BABIES_perfusion'
project_home = '/Users/catcamacho/Dropbox/Projects/infant_ASL/proc'
#project_home = '/Users/axon/Dropbox/infant_ASL/proc'
output_dir = project_home + '/Proc'
wkflow_dir = project_home + '/Workflows'
template = project_home + '/Templates/T2wtemplate_2mm.nii'
mask = project_home + '/Templates/standard_mask.nii'
gm_mask = project_home + '/Templates/gm_mask.nii'

# Files for group level analysis
group_mat = project_home + '/Misc/group_sensitivity_reunion.mat'
t_contrasts = project_home + '/Misc/tcon.con'


# In[ ]:

# Data Handling Nodes

datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir

grabcbfdata = Node(DataGrabber(template=output_dir + '/warped_cbf_vol/*/salsop_cbf_warped.nii', 
                               sort_filelist=True, 
                               outfields=['cbf_list']), 
                   name='grabcbf')


# In[ ]:

# Analysis Nodes

merge = Node(Merge(dimension = 't'), name = 'merge')

apply_mask = Node(ApplyMask(mask_file=mask), name='apply_mask')

randomise = Node(Randomise(tfce = True,
                           num_perm = 500,
                           tcon = t_contrasts,
                           design_mat = group_mat,
                           mask=gm_mask), 
                 name = 'randomise')

threshold_p_file = MapNode(Binarize(min = 0.95), 
                           name = 'threshold_p_file', 
                           iterfield=['in_file'])

mask_t_stat = MapNode(ApplyMask(), 
                      name = 'mask_t_stat',
                      iterfield=['in_file','mask_file'])

cluster = MapNode(Cluster(threshold = 2.2, 
                          out_localmax_txt_file = True, 
                          out_index_file = True), 
                  name = 'cluster', iterfield=['in_file'])


# In[ ]:

# Analysis workflow

grouplevel = Workflow(name='grouplevel')

grouplevel.connect([(grabcbfdata, merge,[('cbf_list', 'in_files')]),
                    (merge, apply_mask, [('merged_file','in_file')]),
                    (apply_mask, randomise, [('out_file', 'in_file')]),
                    (randomise, threshold_p_file, [('t_corrected_p_files', 'in_file')]),
                    (randomise, mask_t_stat, [('tstat_files', 'in_file')]),
                    (threshold_p_file, mask_t_stat, [('binary_file', 'mask_file')]),
                    (mask_t_stat, cluster, [('out_file', 'in_file')]),
                    (cluster, datasink, [('index_file', 'cluster_index_file')]),
                    (cluster, datasink, [('localmax_txt_file', 'cluster_localmax_txt_file')]),
                    (randomise, datasink, [('t_corrected_p_files', 't_corrected_p_files')]),
                    (randomise, datasink, [('tstat_files', 'tstat_files')])
                   ])

grouplevel.base_dir = wkflow_dir
grouplevel.write_graph(graph2use='flat')
grouplevel.run('MultiProc', plugin_args={'n_procs': 2})


# In[ ]:



