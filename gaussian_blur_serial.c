
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PGM_ID_LINE "P5"
#define PGM_ID_LINE_LEN 3
#define PI 3.14159265358979323846
#define NSEC_PER_SEC 1000000000

#define idx(arr, cols, i, j) (arr[(cols)*(i) + (j)])

struct compute_data {
    unsigned char *in_image;
    unsigned char *out_image;
    float *gaussian;
    int width;
    int height;
    int max_pixel_value;
    float sigma;
    int order;
};


/* Offset from a Gaussian's center for x, y at index 0, 0.
 *
 * For example, for order 5, -2 is returned.
 */
static inline float goffset(float order)
{
    return -(order - 1)/2;
}

static inline int clamp(int x, int low, int high)
{
    if (x < low) x = low;
    if (x > high) x = high;
    return x;
}


/* Initialize the Gaussian convolution matrix.
 */
static void init_gaussian(struct compute_data *data)
{
    float sum = 0;
    float x, y, res;
    float offset = goffset(data->order);
    int i, j;

    for (i = 0, y = offset; i < data->order; i++, y++){
        for (j = 0, x = offset; j < data->order; j++, x++){
            res = exp(-(x*x + y*y)/(2*data->sigma*data->sigma));
            //res /= 2*PI*data->sigma*data->sigma;

            sum += res;
            idx(data->gaussian, data->order, i, j) = res;
        }
    }

    for (int i = 0; i < data->order; i++){
        for (int j = 0; j < data->order; j++){
            idx(data->gaussian, data->order, i, j) /= sum;
        }
    }
}


/* Convolute the Gaussian with the input image.
 */
static void apply_gaussian(struct compute_data *data)
{
#ifdef PROGRESS
    float last_perc = -1;
    float next_perc;
#endif

    for (int y = 0; y < data->height; y++) {
        for (int x = 0; x < data->width; x++) {
            float term = 0;
            int xsub, ysub;
            int offset = goffset(data->order);
            int i, j;

            for (i = 0; i < data->order; i++) {
                ysub = clamp(y + offset + i, 0, data->height - 1);

                for (j = 0; j < data->order; j++) {
                    xsub = clamp(x + offset + j, 0, data->width - 1);

                    term += idx(data->in_image, data->width, ysub, xsub)
                        * idx(data->gaussian, data->order, i, j);
                }
            }

            idx(data->out_image, data->width, y, x) = term;
        }

#ifdef PROGRESS
        next_perc = y*100.0/(data->height);

        if (next_perc > last_perc) {
            printf("\rProgress %.1f%%", next_perc);
            fflush(stdout);
            last_perc = next_perc;
        }
#endif
    }

#ifdef PROGRESS
    printf("\n");
#endif
}


// Return time passed in seconds.
static float get_timespec_delta(const struct timespec *start,
        const struct timespec *stop)
{
    long long delta_nsec, start_nsec, stop_nsec;

    start_nsec = start->tv_sec * NSEC_PER_SEC + start->tv_nsec;
    stop_nsec = stop->tv_sec * NSEC_PER_SEC + stop->tv_nsec;
    delta_nsec = stop_nsec - start_nsec;

    return (float)delta_nsec / NSEC_PER_SEC;
}


int main(int argc, char *argv[])
{
    FILE *in_file;
    FILE *out_file;
    int amount, amount_read;
    struct compute_data data;
    struct timespec start, stop;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <sigma>\n", argv[0]);
        return 1;
    }

    in_file = fopen(argv[1], "r");
    if (NULL == in_file) {
        fprintf(stderr, "Error: cannot open file %s\n", argv[1]);
        return 1;
    }

    // Read pgm metadata.
    char id_line[PGM_ID_LINE_LEN + 1];
    if (NULL == fgets(id_line, PGM_ID_LINE_LEN, in_file)
            || strcmp(id_line, PGM_ID_LINE) != 0) {
        fprintf(stderr, "Error: invalid PGM information\n");
        return 1;
    }

    if (fscanf(in_file, "%d %d\n%d\n", &data.width, &data.height,
                &data.max_pixel_value) < 3) {
        fprintf(stderr, "Error: invalid PGM information\n");
        return 1;
    }

    // Read image data.
    amount = data.width*data.height;
    data.in_image = malloc(amount*sizeof(*data.in_image));
    amount_read = fread(data.in_image, sizeof(*data.in_image), amount, in_file);

    if (amount_read < amount) {
        fprintf(stderr, "Error: invalid PGM pixels\n");
        return 1;
    }

    fclose(in_file);


    // Determine sigma and order.
    char *end;
    data.sigma = strtod(argv[3], &end);
    if (end == argv[3] || data.sigma <= 0) {
        fprintf(stderr, "Error: invalid sigma value\n");
        return 1;
    }

    data.order = ceil(6*data.sigma);
    if (data.order % 2 == 0)
        data.order += 1;
    if (data.order > data.width || data.order > data.height) {
        fprintf(stderr, "Error: sigma value too big for image size\n");
        return 1;
    }


    // Compute image
    data.out_image = malloc(amount*sizeof(*data.out_image));
    data.gaussian = malloc(data.order*data.order*sizeof(*data.gaussian));


    clock_gettime(CLOCK_MONOTONIC, &start);

    init_gaussian(&data);

#ifdef DEBUG
    for (int i = 0; i < data.order; i++) {
        for (int j = 0; j < data.order; j++) {
            printf("%f ", idx(data.gaussian, data.order, i, j));
        }
        printf("\n");
    }
#endif

    apply_gaussian(&data);

    clock_gettime(CLOCK_MONOTONIC, &stop);
    printf("Running time: %.6f secs\n", get_timespec_delta(&start, &stop));


    // Output_image
    out_file = fopen(argv[2], "w");
    fprintf(out_file, "%s\n%d %d\n%d\n", PGM_ID_LINE, data.width, data.height,
            data.max_pixel_value);
    fwrite(data.out_image, sizeof(char), amount, out_file);
    fclose(out_file);


    free(data.in_image);
    free(data.out_image);

    return 0;
}
