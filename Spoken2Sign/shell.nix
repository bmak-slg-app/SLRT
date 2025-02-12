with import <nixpkgs> { };
mkShell {
  packages = with pkgs; [
    blender
    ffmpeg
    valkey
  ];
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc.lib
    glib
    zlib
    xorg.libX11
    xorg.libXrender
    xorg.libXxf86vm
    xorg.libXfixes
    xorg.libXi
    xorg.libSM
    xorg.libICE
    libxkbcommon
  ];
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
  SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
}
