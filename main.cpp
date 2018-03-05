//SDFGen - A simple grid-based signed distance field (level set) generator for triangle meshes.
//Written by Christopher Batty (christopherbatty@yahoo.com, www.cs.columbia.edu/~batty)
//...primarily using code from Robert Bridson's website (www.cs.ubc.ca/~rbridson)
//This code is public domain. Feel free to mess with it, let me know if you like it.

#include "makelevelset3.h"
#include "config.h"

#include "hdf5.h"
#include "hdf5_hl.h"
#include "H5Cpp.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

using namespace H5;

int main(int argc, char* argv[]) {
  
  if(argc != 4) {
    std::cout << "SDFGen - A utility for converting closed oriented triangle meshes into grid-based signed distance fields.\n";
    std::cout << "\nThe output file format is:";
    std::cout << "<ni> <nj> <nk>\n";
    std::cout << "<origin_x> <origin_y> <origin_z>\n";
    std::cout << "<dx>\n";
    std::cout << "<value_1> <value_2> <value_3> [...]\n\n";
    
    std::cout << "(ni,nj,nk) are the integer dimensions of the resulting distance field.\n";
    std::cout << "(origin_x,origin_y,origin_z) is the 3D position of the grid origin.\n";
    std::cout << "<dx> is the grid spacing.\n\n";
    std::cout << "<value_n> are the signed distance data values, in ascending order of i, then j, then k.\n";

    std::cout << "The output filename will match that of the input, with the OBJ suffix replaced with SDF.\n\n";

    std::cout << "Usage: SDFGen <filename> <dx> <padding>\n\n";
    std::cout << "Where:\n";
    std::cout << "\t<filename> specifies a Wavefront OBJ (text) file representing a *triangle* mesh (no quad or poly meshes allowed). File must use the suffix \".obj\".\n";
    std::cout << "\t<dx> specifies the length of grid cell in the resulting distance field.\n";
    std::cout << "\t<padding> specifies the number of cells worth of padding between the object bound box and the boundary of the distance field grid. Minimum is 1.\n\n";
    
    exit(-1);
  }

  std::string filename(argv[1]);
  if(filename.size() < 5 || filename.substr(filename.size()-4) != std::string(".obj")) {
    std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
    exit(-1);
  }

  std::stringstream arg2(argv[2]);
  float dx;
  arg2 >> dx;
  
  std::stringstream arg3(argv[3]);
  int padding;
  arg3 >> padding;

  if(padding < 1) padding = 1;
  //start with a massive inside out bound box.
  Vec3f min_box(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()), 
    max_box(-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max());
  
  std::cout << "Reading data.\n";

  std::ifstream infile(argv[1]);
  if(!infile) {
    std::cerr << "Failed to open. Terminating.\n";
    exit(-1);
  }

  int ignored_lines = 0;
  std::string line;
  std::vector<Vec3f> vertList;
  std::vector<Vec3ui> faceList;
  while(!infile.eof()) {
    std::getline(infile, line);

    //.obj files sometimes contain vertex normals indicated by "vn"
    if(line.substr(0,1) == std::string("v") && line.substr(0,2) != std::string("vn")){
      std::stringstream data(line);
      char c;
      Vec3f point;
      data >> c >> point[0] >> point[1] >> point[2];
      vertList.push_back(point);
      update_minmax(point, min_box, max_box);
    }
    else if(line.substr(0,1) == std::string("f")) {
      std::stringstream data(line);
      char c;
      int v0,v1,v2;
      data >> c >> v0 >> v1 >> v2;
      faceList.push_back(Vec3ui(v0-1,v1-1,v2-1));
    }
    else if( line.substr(0,2) == std::string("vn") ){
      std::cerr << "Obj-loader is not able to parse vertex normals, please strip them from the input file. \n";
      exit(-2); 
    }
    else {
      ++ignored_lines; 
    }
  }
  infile.close();
  
  if(ignored_lines > 0)
    std::cout << "Warning: " << ignored_lines << " lines were ignored since they did not contain faces or vertices.\n";

  std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;

  //Add padding around the box.
  Vec3f unit(1,1,1);
  min_box -= padding*dx*unit;
  max_box += padding*dx*unit;
  Vec3ui sizes = Vec3ui((max_box - min_box)/dx);
  
  std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

  std::cout << "Computing signed distance field.\n";
  Array3f phi_grid;
  make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);

  std::string outname;

  // store to HDF file format.
  //Very hackily strip off file suffix.
  outname = filename.substr(0, filename.size()-4) + std::string(".h5");
  std::cout << "Writing results to: " << outname << "\n";

  // Create a new file using the default property lists.
  H5File file(std::string(outname), H5F_ACC_TRUNC);

  /* create a double type dataset named "data" */
  int RANK = 5;
  int chunk_size = 8; // sdf will be stored in chunk_size x chunk_size x chunk_size sized chunks
  hsize_t     dims[RANK]={1,4,chunk_size,chunk_size,chunk_size};
  hsize_t     max_dims[RANK]={H5S_UNLIMITED,4,chunk_size,chunk_size,chunk_size};
  hsize_t     chunk_dims[RANK]={1,4,chunk_size,chunk_size,chunk_size};
  DataSpace *dataspace = new DataSpace(RANK, dims, max_dims);
  // Modify dataset creation property to enable chunking
  DSetCreatPropList prop;
  prop.setChunk(RANK, chunk_dims);
  DataSet *dataset = new DataSet(file.createDataSet("data", PredType::IEEE_F64LE, *dataspace, prop));
  // Dataset Extention details
  hsize_t offset[RANK];
  hsize_t dimsext[RANK]={1,4,chunk_size,chunk_size,chunk_size}; // extend dimensions
  double dataext[1][4][chunk_size][chunk_size][chunk_size];

  std::cout << "no of sdf entries being written: " << phi_grid.a.size() << std::endl;
  std::cout << "voxel grid size: " << phi_grid.ni << " x " << phi_grid.nj << " x " << phi_grid.nk << std::endl;
  std::cout << "chunk grid size: " << phi_grid.ni/chunk_size << " x " << phi_grid.nj/chunk_size << " x " << phi_grid.nk/chunk_size << std::endl;
  int chunk_id = 0;
  for (int k = 0; k < phi_grid.nk/chunk_size; ++k) {
    for (int j = 0; j < phi_grid.nj/chunk_size; ++j) {
      for (int i = 0; i < phi_grid.ni/chunk_size; ++i) {
        // Extend dataset for new block
        if (chunk_id > 0) {
          hsize_t newsize[RANK]={chunk_id+1,4,chunk_size,chunk_size,chunk_size};
          dataset->extend(newsize);
        }

        // fill up chunk
        bool has_surface = false; // flag to indicate if atleast one surface voxel is present in the chunk
        for (int ck = 0; ck < chunk_size; ++ck) for (int cj = 0; cj < chunk_size; ++cj) for (int ci = 0; ci < chunk_size; ++ci) {
          // Fill dataset
          // SDF, global pos. in voxel units (i,j,k)
          dataext[0][0][ck][cj][ci] =  (double)(phi_grid(i*chunk_size+ci, j*chunk_size+cj, k*chunk_size+ck));
          dataext[0][1][ck][cj][ci] =  (double)(i*chunk_size+ci);  // i
          dataext[0][2][ck][cj][ci] =  (double)(j*chunk_size+cj);  // j
          dataext[0][3][ck][cj][ci] =  (double)(k*chunk_size+ck);  // k

          if (abs(phi_grid(i*chunk_size+ci, j*chunk_size+cj, k*chunk_size+ck)) < dx) has_surface = true;
        }
        if (!has_surface) continue;

        // Write data to dataset
        // Select a hyperslab in extended portion of the dataset.
        DataSpace *filespace = new DataSpace(dataset->getSpace());
        offset[0] = chunk_id;
        offset[1] = 0;
        offset[2] = 0;
        offset[3] = 0;
        offset[4] = 0;
        filespace->selectHyperslab(H5S_SELECT_SET, dimsext, offset);

        // Define memory space.
        DataSpace *memspace = new DataSpace(RANK, dimsext, NULL);

        // Write data to the extended portion of the dataset.
        dataset->write(dataext, PredType::IEEE_F64LE, *memspace, *filespace);

        delete filespace;
        delete memspace;

        chunk_id += 1;
      }
    }
  }

  // Close all objects and file.
  prop.close();
  delete dataspace;
  delete dataset;
  file.close();
  std::cout << "In total " << chunk_id << " voxel chunks of size " << chunk_size << " saved to " << outname << std::endl;

  std::cout << "Processing complete.\n";

return 0;
}
