def policy(state,w1,w2):
    
    u,c1 = full_forward_propagation(state, u_params_values, nn_architecture)

    
    v,c2= full_forward_propagation(state,v_params_values,nn_architecture)
    
    
    a=np.random.normal(np.squeeze(u),np.squeeze(v))
    #print(a)
    return a,v,u,c1,c2

def gradients(a,v,u,u_params_values,v_params_values,nn_architecture,c1,c2):
    
    v_grads= v_full_backward_propagation(v, a,u, c2, v_params_values, nn_architecture)
    u_grads= u_full_backward_propagation(u, a, v,c1, u_params_values, nn_architecture)
    #print(v_grads)
    #print(u_grads)
    return v_grads,u_grads

def update_params_with_RMS(params, grads,s, beta, learning_rate):
    
    # grads has the dw and db parameters from backprop
    # params  has the W and b parameters which we have to update 
    for l in range(len(params) // 2 ):
        # HERE WE COMPUTING THE VELOCITIES 
        s["dW" + str(l+1)]= beta * s["dW" + str(l+1)] + (1 - beta) * np.square(grads['dW' + str(l+1)])
        s["db" + str(l+1)] = beta * s["db" + str(l+1)] + (1 - beta) * np.square(grads['db' + str(l+1)])
        
        #updating parameters W and b
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * grads['dW' + str(l+1)] / (np.sqrt( s["dW" + str(l+1)] )+ pow(10,-4))
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * grads['db' + str(l+1)] / (np.sqrt( s["db" + str(l+1)]) + pow(10,-4))

    return params