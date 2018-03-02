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
  int RANK = 2;
  hsize_t     dims[RANK]={1,4};
  hsize_t     max_dims[RANK]={H5S_UNLIMITED,4};
  hsize_t     chunk_dims[RANK]={1,4};
  DataSpace *dataspace = new DataSpace(RANK, dims, max_dims);
  // Modify dataset creation property to enable chunking
  DSetCreatPropList prop;
  prop.setChunk(RANK, chunk_dims);
  DataSet *dataset = new DataSet(file.createDataSet("data", PredType::IEEE_F64LE, *dataspace, prop));
  // Dataset Extention details
  hsize_t offset[RANK];
  hsize_t dimsext[RANK]={1,4}; // extend dimensions
  double dataext[1][4];

  std::cout << "no of sdf entries being written: " << phi_grid.a.size() << std::endl;
  for(unsigned int v = 0; v < phi_grid.a.size(); ++v) {
//    std::cout << "entry: " << v << std::endl;

//    std::cout << "v0" << std::endl;
    // Extend dataset for new block
    if (v > 0) {
      hsize_t newsize[RANK]={v+1,4};
      dataset->extend(newsize);
    }

    // Fill dataset
    // SDF, global pos. in voxel units (i,j,k)
    // v = k*ni*nj + j*ni + i;
    // k = floor(v/(ni*nj))
    // j = floor((v-k*ni*nj)/ni)
    // i = v - k*ni*nj - j*ni
//    std::cout << "v1" << std::endl;
    dataext[0][0] =  (double)(phi_grid.a[v]);
    dataext[0][3] =  (double)(std::floor(v/(phi_grid.ni*phi_grid.nj)));  // k
    dataext[0][2] =  (double)(std::floor((v - dataext[0][3]*phi_grid.ni*phi_grid.nj)/phi_grid.ni)); // j
    dataext[0][1] =  (double)(v - dataext[0][3]*phi_grid.ni*phi_grid.nj - dataext[0][2]*phi_grid.ni); // i

    // Write data to dataset
    // Select a hyperslab in extended portion of the dataset.
//    std::cout << "v2" << std::endl;
    DataSpace *filespace = new DataSpace(dataset->getSpace());
    offset[0] = v;
    offset[1] = 0;
    filespace->selectHyperslab(H5S_SELECT_SET, dimsext, offset);

    // Define memory space.
//    std::cout << "v3" << std::endl;
    DataSpace *memspace = new DataSpace(RANK, dimsext, NULL);

    // Write data to the extended portion of the dataset.
//    std::cout << "v4" << std::endl;
    dataset->write(dataext, PredType::IEEE_F64LE, *memspace, *filespace);

//    std::cout << "v5" << std::endl;
    delete filespace;
    delete memspace;
  }
  // Close all objects and file.
  prop.close();
  delete dataspace;
  delete dataset;
  file.close();


  std::cout << "Processing complete.\n";

return 0;
}
