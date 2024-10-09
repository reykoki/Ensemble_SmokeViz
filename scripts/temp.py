def plot_test_results(data, labels, pred, fn):
    RGB = np.dstack([data[0], data[1], data[2]])
    preds = np.dstack([pred[0], pred[1], pred[2]])
    truths = np.dstack([labels[0], labels[1], labels[2]])
    lat, lon = get_lat_lon(fn)
    num_pixels = data.shape[1]
    X, Y = get_mesh(num_pixels)
    colors = ['red', 'orange', 'yellow']
    fig, ax = plt.subplots(1, 2, figsize=(16,8))

    ax[0].imshow(RGB)
    ax[1].imshow(RGB)
    for idx in range(3):
        ax[0].contour(X,Y,truths[:,:,idx],levels =[.99],colors=[colors[idx]])
        ax[1].contour(X,Y,preds[:,:,idx],levels =[.99],colors=[colors[idx]])

    plt.subplots_adjust(wspace=0)
    plt.show()



def plot_densities_from_processed_data(fn_16, fn_17, data_loc="./sample_data/"):
    RGB_16, truths, lat, lon = get_data(fn_16, data_loc)
    RGB_17, truths, lat, lon = get_data(fn_17, data_loc)

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    ax[0].imshow(RGB_16)
    ax[1].imshow(RGB_17)

    ax[0].set_yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
    ax[0].set_ylabel('latitude (degrees)', fontsize=16)
    ax[0].set_xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
    ax[0].set_xlabel('longitude (degrees)', fontsize=16)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[0].set_title('GOES-16',fontsize=18)
    ax[1].set_title('GOES-17',fontsize=18)
    plt.suptitle(get_datetime_from_fn(fn_16), fontsize=18)

    plt.tight_layout(pad=0)
    plt.savefig('G16_v_G17.png', dpi=300)
    plt.subplots_adjust(wspace=0)
    plt.show()
