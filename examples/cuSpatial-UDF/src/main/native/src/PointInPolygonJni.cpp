/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/polygon_bounding_box.hpp>
#include <cuspatial/shapefile_reader.hpp>
#include <cuspatial/spatial_join.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <jni.h>

namespace {

constexpr char const* RUNTIME_ERROR_CLASS = "java/lang/RuntimeException";
constexpr char const* ILLEGAL_ARG_CLASS   = "java/lang/IllegalArgumentException";

/**
 * @brief Throw a Java exception
 *
 * @param env The Java environment
 * @param class_name The fully qualified Java class name of the exception
 * @param msg The message string to associate with the exception
 */
void throw_java_exception(JNIEnv* env, char const* class_name, char const* msg) {
  jclass ex_class = env->FindClass(class_name);
  if (ex_class != NULL) {
    env->ThrowNew(ex_class, msg);
  }
}

/**
 * @brief check if an java exceptions have been thrown and if so throw a C++
 * exception so the flow control stop processing.
 */
inline void check_java_exception(JNIEnv *const env) {
  if (env->ExceptionCheck()) {
    // Not going to try to get the message out of the Exception, too complex and
    // might fail.
    throw std::runtime_error("JNI Exception...");
  }
}

/** @brief RAII for jstring to be sure it is handled correctly. */
class native_jstring {
private:
  JNIEnv *env;
  jstring orig;
  mutable const char *cstr;
  mutable size_t cstr_length;

  void init_cstr() const {
    if (orig != NULL && cstr == NULL) {
      cstr_length = env->GetStringUTFLength(orig);
      cstr = env->GetStringUTFChars(orig, 0);
      check_java_exception(env);
    }
  }

public:
  native_jstring(native_jstring const &) = delete;
  native_jstring &operator=(native_jstring const &) = delete;

  native_jstring(native_jstring &&other) noexcept
      : env(other.env), orig(other.orig), cstr(other.cstr), cstr_length(other.cstr_length) {
    other.cstr = NULL;
  }

  native_jstring(JNIEnv *const env, jstring orig)
      : env(env), orig(orig), cstr(NULL), cstr_length(0) {}

  native_jstring &operator=(native_jstring const &&other) {
    if (orig != NULL && cstr != NULL) {
      env->ReleaseStringUTFChars(orig, cstr);
    }
    this->env = other.env;
    this->orig = other.orig;
    this->cstr = other.cstr;
    this->cstr_length = other.cstr_length;
    other.cstr = NULL;
    return *this;
  }

  bool is_null() const noexcept { return orig == NULL; }

  const char *get() const {
    init_cstr();
    return cstr;
  }

  ~native_jstring() {
    if (orig != NULL && cstr != NULL) {
      env->ReleaseStringUTFChars(orig, cstr);
    }
  }
};

/**
 * @brief a column is valid only when it has at least one valid row.
 */
inline bool is_invalid_column(cudf::column_view const& col) {
  return col.null_count() == col.size();
}

/**
 * @brief run the reduction 'agg' on the input column 'col'. The input column should be
 * double type, and have at least one valid row. Otherwise, the behavior is undefined.
 */
inline double reduce_as_double(cudf::column_view const& col,
                               std::unique_ptr<cudf::aggregation> const& agg) {
  auto s = cudf::reduce(col, agg, col.type());
  // s is always valid
  auto p_num_scalar = reinterpret_cast<cudf::numeric_scalar<double>*>(s.get());
  return p_num_scalar->value();
}

/**
 * @brief convert the cuspatial result for the 'point_in_polygon' test,
 * to match the simple UDF output requirement.
 *
 * (It is called only once so it can be inline.)
 */
inline jlong convert_point_in_polygon_result(std::unique_ptr<cudf::table> pt_indices,
                                             std::unique_ptr<cudf::column> ply_indices,
                                             cudf::column_view const& raw_xv,
                                             cudf::column_view const& raw_yv) {
  // Sort result by (point_offset, polygon_offset) pair, to make the output deterministic.
  // The sorted polygon index column can be used for the output LIST child column.
  auto sorted_table_ptr = cudf::sort(cudf::table_view({
      pt_indices->get_column(0),
      *ply_indices}));
  // release the GPU resources once they are no longer needed.
  pt_indices.reset();
  ply_indices.reset();

  // Compute the offsets of the output by running the lower_bound with a sequence starting from 0.
  auto raw_pt_indices = cudf::sequence(raw_xv.size() + 1, cudf::numeric_scalar<uint32_t>(0));
  auto offset_col = cudf::lower_bound(
      cudf::table_view({sorted_table_ptr->get_column(0)}),
      cudf::table_view({*raw_pt_indices}),
      {}, {});
  raw_pt_indices.reset();

  // Cast the child column from UINT32 to INT32 (int32 should be enough) for output
  auto child_col = cudf::cast(sorted_table_ptr->get_column(1), cudf::data_type(cudf::type_id::INT32));
  // release the GPU resources once they are no longer needed.
  sorted_table_ptr.reset();

  // Compute the validity mask of the output by 'bitwise-AND'ing the two input point coordinate
  // column validity masks together.
  auto [new_mask, new_null_count] = cudf::bitmask_and(cudf::table_view({raw_xv, raw_yv}));

  // Now all elements are ready, create the final list column and return it.
  auto output_list = cudf::make_lists_column(
      raw_xv.size(),
      std::move(offset_col),
      std::move(child_col),
      new_null_count,
      std::move(new_mask));
  return reinterpret_cast<jlong>(output_list.release());
}

static jlongArray convert_cols_for_return(JNIEnv *env,
                                          std::vector<std::unique_ptr<cudf::column>>& ret) {
  int cols_num = ret.size();
  jlongArray j_out_handles = env->NewLongArray(cols_num);
  jlong* out_handles = env->GetLongArrayElements(j_out_handles, NULL);
  check_java_exception(env);
  if (j_out_handles == nullptr || out_handles == nullptr) {
    throw std::bad_alloc();
  }
  for (int i = 0; i < cols_num; i++) {
    out_handles[i] = reinterpret_cast<jlong>(ret[i].release());
  }
  env->ReleaseLongArrayElements(j_out_handles, out_handles, 0);
  return j_out_handles;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_udf_PointInPolygon_deleteCudfColumn(JNIEnv* env, jclass,
                                                                 jlong j_handle) {
  delete reinterpret_cast<cudf::column *>(j_handle);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_udf_PointInPolygon_readPolygon(JNIEnv* env, jclass,
                                                            jstring j_shape_file) {
  // turn the Java string to the native string.
  // The file is always valid, which is ensured by Java.
  native_jstring shape_file(env, j_shape_file);
  auto poly_cols = cuspatial::read_polygon_shapefile(shape_file.get());
  return convert_cols_for_return(env, poly_cols);
}

/**
 * @brief The native implementation of PointInPolygon.pointInPolygon.
 *
 * @param env The Java environment
 * @param j_x_view The address of the cudf column view of the x column
 * @param j_y_view The address of the cudf column view of the y column
 * @param j_ply_fpos The column address of beginning index of the first ring in each polygon
 * @param j_ply_rpos The column address of beginning index of the first point in each ring
 * @param j_ply_x The column address of x component of polygon points
 * @param j_ply_y The column address of y component of polygon points
 * @return The address of the cudf column containing the results
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_udf_PointInPolygon_pointInPolygon(JNIEnv* env, jclass,
                                                               jlong j_x_view, jlong j_y_view,
                                                               jlong j_ply_fpos, jlong j_ply_rpos,
                                                               jlong j_ply_x, jlong j_ply_y) {
  // Use a try block to translate C++ exceptions into Java exceptions to avoid
  // crashing the JVM if a C++ exception occurs.
  try {
    // turn the addresses into column_view pointers
    auto raw_xv = reinterpret_cast<cudf::column_view const*>(j_x_view);
    auto raw_yv = reinterpret_cast<cudf::column_view const*>(j_y_view);
    if (raw_xv->type().id() != raw_yv->type().id()) {
      throw_java_exception(env, ILLEGAL_ARG_CLASS, "x and y have different types");
      return 0;
    }

    // The python test casts the data to numpy.float32, but here cudf FLOAT32 does not work.
    // It complains the error
    //    " quadtree_point_in_polygon.cu:275: points and polygons must have the same data type"
    //
    // so convert int32 to cudf FLOAT64 and it works.
    auto xv = cudf::cast(*raw_xv, cudf::data_type(cudf::type_id::FLOAT64));
    auto yv = cudf::cast(*raw_yv, cudf::data_type(cudf::type_id::FLOAT64));

    // check shape data
    auto ply_fpos = reinterpret_cast<cudf::column_view const*>(j_ply_fpos);
    auto ply_rpos = reinterpret_cast<cudf::column_view const*>(j_ply_rpos);
    auto ply_x = reinterpret_cast<cudf::column_view const*>(j_ply_x);
    auto ply_y = reinterpret_cast<cudf::column_view const*>(j_ply_y);

    if (is_invalid_column(*ply_x) || is_invalid_column(*ply_y)) {
      // No polygon data, then return a list column of all nulls.
      // The offsets are all 0, and the child is an empty column.
      auto offset_col = cudf::make_column_from_scalar(cudf::numeric_scalar<int32_t>(0),
          xv->size() + 1);
      auto nulls_list = cudf::make_lists_column(
          xv->size(),
          std::move(offset_col),
          cudf::make_empty_column(cudf::type_id::INT32),
          xv->size(),
          cudf::create_null_mask(xv->size(), cudf::mask_state::ALL_NULL));
      return reinterpret_cast<jlong>(nulls_list.release());
    }

    auto min_agg = cudf::make_min_aggregation();
    auto max_agg = cudf::make_max_aggregation();
    auto x_min = reduce_as_double(*ply_x, min_agg);
    auto x_max = reduce_as_double(*ply_x, max_agg);
    auto y_min = reduce_as_double(*ply_y, min_agg);
    auto y_max = reduce_as_double(*ply_y, max_agg);

    // 2) quadtree construction
    cudf::size_type min_size = 512;
    int8_t num_levels = 15;
    double scale = std::max(std::abs(x_max - x_min), std::abs(y_max - y_min)) / ((1 << num_levels) - 2);

    auto [point_indices, quadtree] = cuspatial::quadtree_on_points(
        *xv, *yv, x_min, x_max, y_min, y_max, scale, num_levels, min_size);

    // 3) run the computation
    auto poly_bboxes = cuspatial::polygon_bounding_boxes(*ply_fpos, *ply_rpos, *ply_x, *ply_y);

    auto poly_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
        *quadtree, *poly_bboxes, x_min, x_max, y_min, y_max, scale, num_levels);
    // release the GPU resources once they are no longer needed.
    poly_bboxes.reset();

    auto point_in_polygon_pairs = cuspatial::quadtree_point_in_polygon(
        *poly_quadrant_pairs, *quadtree, *point_indices,
        *xv, *yv,
        *ply_fpos, *ply_rpos, *ply_x, *ply_y);
    // release the GPU resources once they are no longer needed.
    poly_quadrant_pairs.reset();
    quadtree.reset();

    // The result table has two columns, where each row represents a
    // (polygon_offset, point_offset) pair:
    //    - polygon_offset - UINT32 column of polygon indices
    //    - point_offset   - UINT32 column of point indices
    auto ply_pt_offset = point_in_polygon_pairs->release();

    // 4) convert the `point_offset` to indices of the input points
    auto input_pt_indices = cudf::gather(
        cudf::table_view({*point_indices}),
        *(ply_pt_offset[1]));
    // release the GPU resources once they are no longer needed.
    point_indices.reset();
    ply_pt_offset[1].reset();

    // 4) convert the cuspatail result and return.
    return convert_point_in_polygon_result(
        std::move(input_pt_indices), std::move(ply_pt_offset[0]), *xv, *yv);

  } catch (std::bad_alloc const& e) {
    auto msg = std::string("Unable to allocate native memory: ") +
        (e.what() == nullptr ? "" : e.what());
    throw_java_exception(env, RUNTIME_ERROR_CLASS, msg.c_str());

  } catch (std::invalid_argument const& e) {
    auto msg = e.what() == nullptr ? "" : e.what();
    throw_java_exception(env, ILLEGAL_ARG_CLASS, msg);

  } catch (std::exception const& e) {
    auto msg = e.what() == nullptr ? "" : e.what();
    throw_java_exception(env, RUNTIME_ERROR_CLASS, msg);
  }
  return 0;
}

}
