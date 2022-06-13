/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <TinyAD/Utils/Out.hh>
#include <TinyAD-Examples/Filesystem.hh>

/**
 * Composes *.png files in _path to video.
 * Calls ffmpeg via command line.
 */
inline void compose_video(
        const fs::path& _path,
        const int _fps,
        const std::string& _filename)
{
    // ffmpeg -r 30 -f image2 -i %06d.png -vcodec libx264 -pix_fmt yuv420p -y animation.mp4

    // Convert png files to mp4
    const auto ffmpeg = "ffmpeg -r " + std::to_string(_fps) + " -f image2 -i "
            + (_path / "%06d.png").string()
            + " -vcodec libx264 -pix_fmt yuv420p -y "
            + (_path / _filename).string();

    TINYAD_INFO(ffmpeg);
    std::system(ffmpeg.c_str());
}
