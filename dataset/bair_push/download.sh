if [ ! -f "bair_robot_pushing_dataset_v0.tar" ]; then
  wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
fi
tar -xvf bair_robot_pushing_dataset_v0.tar
mv softmotion30_44k tfrecords
mkdir tfrecords/val
mv tfrecords/train/traj_256_to_511.tfrecords tfrecords/val/