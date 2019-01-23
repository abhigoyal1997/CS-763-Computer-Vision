function transformations = composedTransforms(rotations, pose, adj_bones, num_bones, bones, transformations, root)
    x = [pose(root,:)'; 1];
    for i=1:num_bones(root)
        idx = adj_bones(root,i);
        next_joint = bones(idx,1);
        p_translation = eye(4);
        n_translation = eye(4);
        p_translation(1:3,end) = x(1:3);
        n_translation(1:3,end) = -x(1:3);
        rotation = [squeeze(rotations(idx,:,:)),zeros(3,1); 0 0 0 1];
        transformations(:,:,next_joint) = transformations(:,:,root)*p_translation*rotation*n_translation;
        transformations = composedTransforms(rotations, pose, adj_bones, num_bones, bones, transformations, next_joint);
    end
end

