% select rolling window length to use - an optimisable parameter via pso?
rolling_window_length = 50 ;
batchsize = 5 ;

%  how-many timesteps do we look back for directed connections - this is what we call the "order" of the model 
n1 = 3 ; % first "gaussian" layer order, a best guess just for batchdata creation purposes
n2 = 3 ; % second "binary" layer order, a best guess just for batchdata creation purposes

%  taking into account rolling_window_length, n1, n2 and batchsize, get total lookback length
remainder = rem( ( rolling_window_length + n1 + n2 ) , batchsize ) ;

if ( remainder > 0 ) % number of training examples with lookback and orders n1 and n2 not exactly divisable by batchsize
lookback_length = ( rolling_window_length + n1 + n2 + ( batchsize - remainder ) ) ; % increase the lookback_length
else                 % number of training examples with lookback and orders n1 and n2 exactly divisable by batchsize
lookback_length = ( rolling_window_length + n1 + n2 ) ;
end

%  create batchdataindex using lookback_length to index bars in the features matrix
batchdataindex = ( ( training_point_index - ( lookback_length - 1 ) ) : 1 : training_point_index )' ;
batchdata = features( batchdataindex , : ) ;

%  now that the batchdata has been created, check it for autocorrelation in the features
all_ar_coeff = zeros( size( batchdata , 2 ) , 1 ) ;

  for ii = 1 : size( batchdata , 2 )
  ar_coeffs = arburg( batchdata( : , ii ) , 10 , 'FPE' ) ;
  all_ar_coeff( ii ) = length( ar_coeffs ) - 1 ;
  end
  
%  set order of gaussian_crbm, n1, to be equal to the average length of any autocorrelation in the data
n1 = round( mean( all_ar_coeff ) ) ;  

%  z-normalise the batchdata matrix with the mean and std of columns 
data_mean = mean( batchdata , 1 ) ;
data_std = std( batchdata , 1 ) ;
batchdata = ( batchdata - repmat( data_mean , size( batchdata , 1 ) , 1 ) ) ./ repmat( data_std , size( batchdata , 1 ) , 1 ) ; % batchdata is now z-normalised by data_mean & data_std

%  create the minibatch index matrix for gaussian rbm pre-training of directed weights w
minibatch = ( 1 : 1 : size( batchdata , 1 ) ) ; minibatch( 1 : ( size( batchdata , 1 ) - rolling_window_length ) ) = [] ;
minibatch = minibatch( randperm( size( minibatch , 2 ) ) ) ; minibatch = reshape( minibatch , batchsize , size( minibatch , 2 ) / batchsize ) ; 

% PRE-TRAINING FOR THE VISABLE TO HIDDEN AND THE VISIBLE TO VISIBLE WEIGHTS %%%%
% First create a training set and target set for the pre-training of gaussian layer
dAuto_Encode_targets = batchdata ; dAuto_Encode_training_data = [] ;
% dAuto_Encode_targets = batchdata( : , 2 : end ) ; dAuto_Encode_training_data = [] ; % if bias added to raw data
  
  % loop to create the dAuto_Encode_training_data ( n1 == "order" of the gaussian layer of crbm )
  for ii = 1 : n1
  dAuto_Encode_training_data = [ dAuto_Encode_training_data shift( batchdata , ii ) ] ;
  end

% now delete the first n1 rows due to circular shift induced mismatch of data and targets
dAuto_Encode_targets( 1 : n1 , : ) = [] ; dAuto_Encode_training_data( 1 : n1 , : ) = [] ;

% DO RBM PRE-TRAINING FOR THE BOTTOM UP DIRECTED WEIGHTS W %%%%%%%%%%%%%%%%%%%%%
% use rbm trained initial weights instead of using random initialisation for weights
% Doing this because we are not using regularisation in the autoencoder pre-training
epochs = 10000 ;
hidden_layer_size = 4 * size( dAuto_Encode_targets , 2 ) ;
[ w_weights , w_weights_hid_bias , w_weights_vis_bias ] = cc_gaussian_rbm( dAuto_Encode_targets , minibatch , epochs , hidden_layer_size , 0.05 ) ;
% keep a copy of these original w_weights
w1 = w_weights ;
[ A_weights , A_weights_hid_bias , A_weights_vis_bias ] = cc_gaussian_rbm( dAuto_Encode_training_data , minibatch , epochs , size( dAuto_Encode_targets , 2 ) , 0.05 ) ;
[ B_weights , B_weights_hid_bias , B_weights_vis_bias ] = cc_gaussian_rbm( dAuto_Encode_training_data , minibatch , epochs , hidden_layer_size , 0.05 ) ;

% END OF RBM PRE-TRAINING OF AUTOENCODER WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1) ; surf( A_weights ) ; title( 'A Weights after RBM training' ) ;
figure(2) ; surf( B_weights ) ; title( 'B Weights after RBM training' ) ;
figure(3) ; surf( w_weights ) ; title( 'w Weights after RBM training' ) ;
figure(4) ; plot( A_weights_hid_bias , 'b' , B_weights_hid_bias , 'r' , w_weights_vis_bias , 'g' ) ; title( 'Biases after RBM training' ) ; legend( 'A' , 'B' , 'w' ) ;

% DO THE AUTOENCODER TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create weight update matrices
A_weights_update = zeros( size( A_weights ) ) ;
A_weights_hid_bias_update = zeros( size( A_weights_hid_bias ) ) ;
B_weights_update = zeros( size( B_weights ) ) ;
B_weights_hid_bias_update = zeros( size( B_weights_hid_bias ) ) ;
w_weights_update = zeros( size( w_weights ) ) ;
w_weights_vis_bias_update = zeros( size( w_weights_vis_bias ) ) ;

% for adagrad
historical_A = zeros( size( A_weights ) ) ;
historical_A_hid_bias = zeros( size( A_weights_hid_bias ) ) ;
historical_B = zeros( size( B_weights ) ) ;
historical_B_hid_bias = zeros( size( B_weights_hid_bias ) ) ;
historical_w = zeros( size( w_weights ) ) ;
historical_w_vis_bias = zeros( size( w_weights_vis_bias ) ) ;

% set some training parameters
n = size( dAuto_Encode_training_data , 1 ) ; % number of training examples in dAuto_Encode_training_data
input_layer_size = size( dAuto_Encode_training_data , 2 ) ;
fudge_factor = 1e-6 ; % for numerical stability for adagrad
learning_rate = 0.01 ; % will be changed to 0.001 after 50 iters through epoch loop
mom = 0 ;            % will be changed to 0.9 after 50 iters through epoch loop
noise = 0.5 ;
epochs = 1000 ;
cost = zeros( epochs , 1 ) ;
lowest_cost = inf ;

  % Stochastic Gradient Descent training over dAuto_Encode_training_data 
  for iter = 1 : epochs
   
      % change momentum and learning_rate after 50 iters
      if iter == 50
      mom = 0.9 ;
      learning_rate = 0.001 ;
      end
  
      index = randperm( n ) ; % randomise the order of training examples
     
      for training_example = 1 : n
      
      % Select data for this training batch
      tmp_X = dAuto_Encode_training_data( index( training_example ) , : ) ;
      tmp_T = dAuto_Encode_targets( index( training_example ) , : ) ;
      
      % Randomly black out some of the input training data
      tmp_X( rand( size( tmp_X ) ) < noise ) = 0 ;
      
      % feedforward tmp_X through B_weights and get sigmoid e.g ret = 1.0 ./ ( 1.0 + exp(-input) )
      tmp_X_through_sigmoid = 1.0 ./ ( 1.0 + exp( - ( tmp_X * B_weights + B_weights_hid_bias ) ) ) ;
      
      % Randomly black out some of tmp_X_through_sigmoid for dropout training
      tmp_X_through_sigmoid( rand( size( tmp_X_through_sigmoid ) ) < noise ) = 0 ;
    
      % feedforward tmp_X through A_weights and add to tmp_X_through_sigmoid * w_weights for linear output layer
      final_output_layer = ( tmp_X * A_weights + A_weights_hid_bias ) + ( tmp_X_through_sigmoid * w_weights' + w_weights_vis_bias ) ;
    
      % now do backpropagation
      % this is the derivative of weights for the linear final_output_layer
      delta_out = ( tmp_T - final_output_layer ) ;
      
      % NOTE! gradient of sigmoid function g = sigmoid(z) .* ( 1.0 - sigmoid(z) )
      sig_grad = tmp_X_through_sigmoid .* ( 1 - tmp_X_through_sigmoid ) ; 
      
      % backpropagation only through the w_weights that are connected to tmp_X_through_sigmoid
      delta_hidden = ( delta_out * w_weights ) .* sig_grad ;
      
      % apply deltas from backpropagation with adagrad for the weight updates
      historical_A = historical_A + ( tmp_X' * delta_out ).^2 ;    
      A_weights_update = mom .* A_weights_update + ( learning_rate .* ( tmp_X' * delta_out ) ) ./ ( fudge_factor + sqrt( historical_A ) ) ;
      
      historical_A_hid_bias = historical_A_hid_bias + delta_out.^2 ;
      A_weights_hid_bias_update = mom .* A_weights_hid_bias_update + ( learning_rate .* delta_out ) ./ ( fudge_factor + sqrt( historical_A_hid_bias ) ) ;
      
      historical_w = historical_w + ( delta_out' * tmp_X_through_sigmoid ).^2 ;
      w_weights_update = mom .* w_weights_update + ( learning_rate .* ( delta_out' * tmp_X_through_sigmoid ) ) ./ ( fudge_factor + sqrt( historical_w ) ) ;
      
      historical_w_vis_bias = historical_w_vis_bias + delta_out.^2 ;
      w_weights_vis_bias_update = mom .* w_weights_vis_bias_update + ( learning_rate .* delta_out ) ./ ( fudge_factor + sqrt( historical_w_vis_bias ) ) ;
      
      historical_B = historical_B + ( tmp_X' * delta_hidden ).^2 ;
      B_weights_update = mom .* B_weights_update + ( learning_rate .* ( tmp_X' * delta_hidden ) ) ./ ( fudge_factor + sqrt( historical_B ) ) ;
      
      historical_B_hid_bias = historical_B_hid_bias + delta_hidden.^2 ;
      B_weights_hid_bias_update = mom .* B_weights_hid_bias_update + ( learning_rate .* delta_hidden ) ./ ( fudge_factor + sqrt( historical_B_hid_bias ) ) ;
      
      % update the weight matrices with weight_updates
      A_weights = A_weights + A_weights_update ;
      A_weights_hid_bias = A_weights_hid_bias + A_weights_hid_bias_update ;
      B_weights = B_weights + B_weights_update ;
      B_weights_hid_bias = B_weights_hid_bias + B_weights_hid_bias_update ;
      w_weights = w_weights + w_weights_update ;
      w_weights_vis_bias = w_weights_vis_bias + w_weights_vis_bias_update ;
      
      end % end of training_example loop
  
  % feedforward with this epoch's updated weights
  epoch_trained_tmp_X_through_sigmoid = 1.0 ./ ( 1.0 + exp( -( dAuto_Encode_training_data * B_weights + repmat( B_weights_hid_bias , size( dAuto_Encode_training_data , 1 ) , 1 ) ) ) ) ;
  epoch_trained_output = ( dAuto_Encode_training_data * A_weights + repmat( A_weights_hid_bias , size( dAuto_Encode_training_data , 1 ) , 1 ) )...
                          + ( epoch_trained_tmp_X_through_sigmoid * w_weights' + repmat( w_weights_vis_bias , size( epoch_trained_tmp_X_through_sigmoid , 1 ) , 1 ) ) ;
 
  % get sum squared error cost
  cost( iter , 1 ) = sum( sum( ( dAuto_Encode_targets - epoch_trained_output ) .^ 2 ) ) ;
  
    % record best so far
    if cost( iter , 1 ) <= lowest_cost
       lowest_cost = cost( iter , 1 ) ;
       iter_min = iter ;
       best_A = A_weights ;
       best_B = B_weights ;
       best_w = w_weights ;
    end
  
  end % end of backpropagation epoch loop

% plot weights
figure(5) ; surf( best_A ) ; title( 'Best A Weights' ) ;
figure(6) ; surf( best_B ) ; title( 'Best B Weights' ) ;
figure(7) ; surf( best_w ) ; title( 'Best w Weights' ) ;
figure(8) ; plot( A_weights_hid_bias , 'b' , B_weights_hid_bias , 'r' , w_weights_vis_bias , 'g' ) ; title( 'Biases after Autoencoder training' ) ; legend( 'A' , 'B' , 'w' ) ;
figure(9) ; plot( cost ) ; title( 'Evolution of Autoencoder cost' ) ;

% END OF CRBM WEIGHT PRE-TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% PRE-TRAINING FOR THE VISABLE TO HIDDEN AND THE VISIBLE TO VISIBLE WEIGHTS %%%%
% First create a training set and target set for the pre-training

dAuto_Encode_targets = batchdata ; 
dAuto_Encode_training_data = [] ;

  
  % loop to create the dAuto_Encode_training_data ( n1 == "order" of the gaussian layer of crbm )
  n1 = 3;
  for ii = 1 : n1
  dAuto_Encode_training_data = [ dAuto_Encode_training_data shift( batchdata , ii ) ] ;
  end

% now delete the first n1 rows due to circular shift induced mismatch of data and targets
dAuto_Encode_targets( 1 : n1 , : ) = [] ; dAuto_Encode_training_data( 1 : n1 , : ) = [] ; 

% DO RBM PRE-TRAINING FOR THE BOTTOM UP DIRECTED WEIGHTS W %%%%%%%%%%%%%%%%%%%%%
% use rbm trained initial weights instead of using random initialisation for weights
% Doing this because we are not using regularisation in the autoencoder pre-training
epochs = 5000 ;
hidden_layer_size = 2 * size( dAuto_Encode_targets , 2 ) ;
w_weights = gaussian_rbm( dAuto_Encode_targets , minibatch , epochs , hidden_layer_size ) ;
A_weights = gaussian_rbm( dAuto_Encode_training_data , minibatch , epochs , size( dAuto_Encode_targets , 2 ) ) ;
B_weights = gaussian_rbm( dAuto_Encode_training_data , minibatch , epochs , hidden_layer_size ) ;

% create weight update matrices
A_weights_update = zeros( size( A_weights ) ) ;
B_weights_update = zeros( size( B_weights ) ) ;
w_weights_update = zeros( size( w_weights ) ) ;

% for adagrad
historical_A = zeros( size( A_weights ) ) ;
historical_B = zeros( size( B_weights ) ) ;
historical_w = zeros( size( w_weights ) ) ;

% set some training parameters
n = size( dAuto_Encode_training_data , 1 ) ; % number of training examples in dAuto_Encode_training_data
input_layer_size = size( dAuto_Encode_training_data , 2 ) ;
fudge_factor = 1e-6 ; % for numerical stability for adagrad
learning_rate = 0.1 ; % will be changed to 0.01 after 50 iters through epoch loop
mom = 0 ;             % will be changed to 0.9 after 50 iters through epoch loop
noise = 0.5 ;
epochs = 1000 ;
cost = zeros( epochs , 1 ) ;
lowest_cost = inf ;

  % Stochastic Gradient Descent training over dAuto_Encode_training_data 
  for iter = 1 : epochs
   
      % change momentum and learning_rate after 50 iters
      if iter == 50
      mom = 0.9 ;
      learning_rate = 0.01 ;
      end
  
      index = randperm( n ) ; % randomise the order of training examples
     
      for training_example = 1 : n
      
      % Select data for this training batch
      tmp_X = dAuto_Encode_training_data( index( training_example ) , : ) ;
      tmp_T = dAuto_Encode_targets( index( training_example ) , : ) ;
      
      % Randomly black out some of the input training data
      tmp_X( rand( size( tmp_X ) ) < noise ) = 0 ;
      
      % feedforward tmp_X through B_weights and get sigmoid e.g ret = 1.0 ./ ( 1.0 + exp(-input) )
      tmp_X_through_sigmoid = 1.0 ./ ( 1.0 + exp( - B_weights * tmp_X' ) ) ;
      
      % Randomly black out some of tmp_X_through_sigmoid for dropout training
      tmp_X_through_sigmoid( rand( size( tmp_X_through_sigmoid ) ) < noise ) = 0 ;
    
      % feedforward tmp_X through A_weights and add to tmp_X_through_sigmoid * w_weights for linear output layer
      final_output_layer = ( tmp_X * A_weights' ) + ( tmp_X_through_sigmoid' * w_weights ) ;
    
      % now do backpropagation
      % this is the derivative of weights for the linear final_output_layer
      delta_out = ( tmp_T - final_output_layer ) ;
      
      % NOTE! gradient of sigmoid function g = sigmoid(z) .* ( 1.0 - sigmoid(z) )
      sig_grad = tmp_X_through_sigmoid .* ( 1 - tmp_X_through_sigmoid ) ; 
      
      % backpropagation only through the w_weights that are connected to tmp_X_through_sigmoid
      delta_hidden = ( w_weights * delta_out' ) .* sig_grad ;
      
      % apply deltas from backpropagation with adagrad for the weight updates
      historical_A = historical_A + ( delta_out' * tmp_X ).^2 ;    
      A_weights_update = mom .* A_weights_update + ( learning_rate .* ( delta_out' * tmp_X ) ) ./ ( fudge_factor + sqrt( historical_A ) ) ;
      
      historical_w = historical_w + ( tmp_X_through_sigmoid * delta_out ).^2 ;
      w_weights_update = mom .* w_weights_update + ( learning_rate .* ( tmp_X_through_sigmoid * delta_out ) ) ./ ( fudge_factor + sqrt( historical_w ) ) ;
      
      historical_B = historical_B + ( delta_hidden * tmp_X ).^2 ;
      B_weights_update = mom .* B_weights_update + ( learning_rate .* ( delta_hidden * tmp_X ) ) ./ ( fudge_factor + sqrt( historical_B ) ) ;
      
      % update the weight matrices with weight_updates
      A_weights = A_weights + A_weights_update ;
      B_weights = B_weights + B_weights_update ;
      w_weights = w_weights + w_weights_update ;
      
      end % end of training_example loop
  
  % feedforward with this epoch's updated weights
  epoch_trained_tmp_X_through_sigmoid = ( 1.0 ./ ( 1.0 + exp( -( ( B_weights./2 ) * dAuto_Encode_training_data' ) ) ) ) ;
  epoch_trained_output = ( dAuto_Encode_training_data * ( A_weights./2 )' ) + ( epoch_trained_tmp_X_through_sigmoid' * ( w_weights./2 ) ) ;
 
  % get sum squared error cost
  cost( iter , 1 ) = sum( sum( ( dAuto_Encode_targets - epoch_trained_output ) .^ 2 ) ) ;
  
    % record best so far
    if cost( iter , 1 ) <= lowest_cost
       lowest_cost = cost( iter , 1 ) ;
       best_A = A_weights ./ 2 ;
       best_B = B_weights ./ 2 ;
       best_w = w_weights ./ 2;
    end
  
  end % end of backpropagation loop

lowest_cost                                        % print final cost to terminal
graphics_toolkit( 'qt' ) ;
figure(1) ; plot( cost , 'r' , 'linewidth' , 3 ) ; % and plot the cost curve

% plot weights
graphics_toolkit( 'gnuplot' ) ;
figure(2) ; surf( best_A ) ; title( 'Best A Weights' ) ;
figure(3) ; surf( best_B ) ; title( 'Best B Weights' ) ;
figure(4) ; surf( best_w ) ; title( 'Best w Weights' ) ;

% END OF CRBM WEIGHT PRE-TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




