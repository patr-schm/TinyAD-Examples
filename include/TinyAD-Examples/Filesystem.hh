/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <TinyAD/Utils/Out.hh>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::experimental::filesystem;

inline fs::path SOURCE_PATH = fs::path(SOURCE_PATH_STR);
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);
inline fs::path OUTPUT_PATH = fs::path(OUTPUT_PATH_STR);

inline void make_file_directory(
        const std::string& _file_path)
{
    fs::create_directories(fs::path(_file_path).parent_path());
}

inline void append_to_file(
        const fs::path& _file_path,
        const std::string& _line)
{
    make_file_directory(_file_path);

    std::ofstream file(_file_path, std::ofstream::app);
    TINYAD_ASSERT(file.good());

    file << _line << std::endl;

    TINYAD_INFO("Wrote (append) " << _file_path);
}

template<typename Arg0, typename ...Args>
void append_to_csv(
        const fs::path& _file_path,
        Arg0 arg0,
        Args ...args)
{
    std::stringstream s;
    s << arg0;
    ((s << ", " << args), ...);
    append_to_file(_file_path, s.str());
}

inline std::string pad_integer(
        const int _i,
        const int _pad = 6)
{
    std::ostringstream s;
    s << std::setw(_pad) << std::setfill('0') << _i;

    return s.str();
}

inline std::string int_filename(
        const int _i,
        const int _pad = 6,
        const std::string& _extension = ".png")
{
    return pad_integer(_i, _pad) + _extension;
}
