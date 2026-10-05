# athena-ONNX

If setupATLAS does not work out of the box try including the following in your .bashrc file:

>export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
>alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'

## MNIST Handwritten Digit Classification

In lxplus create a workspace (e.g. mkdir onnxTutorial)

>setupATLAS
>asetup Athena, master, latest
>mkdir build run
>lsetup git
>git atlas init-workdir https://:@gitlab.cern.ch:8443/atlas/athena.git
>cd athena
>git checkout -b onnx
>git atlas addpkg AthExOnnxRuntime
>cd ../build
>cmake ../athena/Projects/WorkDir
>make
>source x86_64-centos7-gcc11-opt/setup.sh
>cd ../run
>cp ../athena/Control/AthenaExamples/AthExOnnxRuntime/share/AthExOnnxRuntime_jobOptions.py .
>athena AthExOnnxRuntime_jobOptions.py


## Topocluster calibration

Log out of lxplus and log back in
In lxplus create a new workspace (eg. 'onnxTopoclusterTutorial')

>mkdir onnxTopoclusterTutorial
>cd onnxTopoclusterTutorial
>setupATLAS
>asetup Athena, master, latest
>mkdir build run
>lsetup git
>git atlas init-workdir https://gitlab.cern.ch/dhangal/athena.git
>cd athena
>git checkout onnx_ml
>git atlas addpkg AthExOnnxRuntime
>cd ../build
>cmake ../athena/Projects/WorkDir
>make
>source x86_64-centos7-gcc11-opt/setup.sh
>cd ../run
>cp ../athena/Control/AthenaExamples/AthExOnnxRuntime/share/AthExOnnxRuntime_jobOptions.py .

Change the onnx model file and input root file in the job options file file 'AthExOnnxRuntime_jobOptions.py' to match the ones shown at the end of this page.

>athena AthExOnnxRuntime_jobOptions.py

input root file : /afs/cern.ch/work/d/dhangal/public/pi0_fully_processed_wFolds.root
onnx model file : /afs/cern.ch/user/d/dhangal/public/model_fold0_opset10.onnx
