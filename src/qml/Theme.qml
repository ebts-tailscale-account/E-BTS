import QtQuick 2.12

// Shared palette for the plain (non-Controls) rectangles/labels drawn across
// the pane/ribbon QML files. QtQuick Controls 2 elements get their green
// Material accent from Material.accent/Material.theme set once on the
// ApplicationWindow in Main.qml (Controls attached properties are inherited
// down the visual tree automatically, no plumbing needed for those).
QtObject {
    readonly property color accent: "#43A047" // Material Green 600
    readonly property color primary: "#1B5E20" // Material Green 900
    readonly property color background: "#121212"
    readonly property color surface: "#1E1E1E"
    readonly property color surfaceElevated: "#262626"
    readonly property color text: "#EAEAEA"
    readonly property color mutedText: "#9E9E9E"
    readonly property color danger: "#E53935"
}
