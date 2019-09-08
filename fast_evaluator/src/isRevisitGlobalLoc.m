function [is_revisit, min_dist] = isRevisitGlobalLoc(query_pose, db_poses, thres)

num_dbs = size(db_poses, 1);  % num_dbs = length(db_poses); this will get 2 when db_poses size is (1,2); MATLAB R2017b

dists = zeros(1, num_dbs);
for ii=1:num_dbs
    dist = norm(query_pose - db_poses(ii, :));
    dists(ii) = dist;    
end

if ( min(dists) < thres ) 
    is_revisit = 1;
else
    is_revisit = 0;
end

min_dist = min(dists);

end

