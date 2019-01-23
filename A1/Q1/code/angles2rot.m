function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    
    n = size(angles_list,1);
    
    cosa = cosd(angles_list);
    sina = sind(angles_list);
    
    rot_matrix = zeros(n,3,3);
    for i=1:n
        rx = [1 0 0; 0 cosa(i,1) -sina(i,1); 0 sina(i,1) cosa(i,1)];
        ry = [cosa(i,2) 0 sina(i,2); 0 1 0; -sina(i,2) 0 cosa(i,2)];
        rz = [cosa(i,3) -sina(i,3) 0; sina(i,3) cosa(i,3) 0; 0 0 1];
        rot_matrix(i,:,:) = rz*ry*rx;
    end
end