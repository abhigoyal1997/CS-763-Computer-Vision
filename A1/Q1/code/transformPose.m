function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here 
    
    nj = size(pose,1);
    nb = size(kinematic_chain,1);
    adj_bones = zeros(nj);
    num_bones = zeros(nj,1);
    
    for i=1:nb
        s = kinematic_chain(i,2);
        num_bones(s) = num_bones(s)+1;
        adj_bones(s,num_bones(s)) = i;
    end
    
    composed_rot = repmat(eye(4),[1,1,nj]);
    composed_rot = composedTransforms(rotations, pose, adj_bones, num_bones, kinematic_chain, composed_rot, root_location);
    
    for i=1:nj
        x = [pose(i,:)';1];
        x = composed_rot(:,:,i)*x;
        pose(i,:) = x(1:3)';
    end
    result_pose = pose;
end